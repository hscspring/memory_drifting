from typing import Dict, Any
import os
from datetime import datetime

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast

from transformers import AutoTokenizer
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.tensorboard import SummaryWriter


from model.kv import CASSKVInjectionModel
from dataset.kv import CASSKVSequentialCollator
from model_utils import save_checkpoint



log_dir = os.path.join(
    "tensorboard_log/",
    datetime.now().strftime("%Y%m%d-%H%M%S") + "grok2"
)

writer = SummaryWriter(log_dir=log_dir)



def train_one_dialogue(model: CASSKVInjectionModel, batch: Dict[str, Any], optimizer) -> tuple[Dict, Dict]:
    """
    训练一个完整对话（所有 turns）
    batch["dialogue_turns"] 来自你的 CASSKVSequentialCollator
    """
    device = model.device
    
    # 初始化记忆状态
    m_prev = model.m_0.detach().clone().to(device)          # (1, history_len, hidden)
    current_len = torch.tensor([1], device=device)          # 初始只有最新槽位（空状态）
    
    total_loss = 0.0
    loss_records = []  # 记录每 turn 的各项 loss
    
    # 用于监控指标
    sims_active = []    # update turn 中 M_new[-1] 与权威 target 的余弦相似度
    gates_active = []   # update turn 中 probe 预测的 gate 值（sigmoid后）
    gates_pred_active = []  # probe 实际预测的概率（用于看是否学得准）

    for _turn_id, turn in enumerate(batch["dialogue_turns"]):
        input_ids = turn["input_ids"].unsqueeze(0).to(device)      # (1, seq_len)
        labels = turn["labels"].unsqueeze(0).to(device)            # (1, seq_len)
        update_flag = turn["update_flag"].to(device)               # (1,)
        turn_id = turn["turn_id"].to(device)                       # (1,)
        print(input_ids.shape, labels.shape, update_flag.shape, turn_id.shape)

        # 准备监督信号
        if update_flag.item() > 0.5:  # update turn
            slot_label = None
            target_ids = turn["target_ids"].unsqueeze(0).to(device)
            negative_state_ids = turn["negative_state_ids"].to(device)
            with torch.no_grad():
                # 1, H
                e_target = model.base_model.model(target_ids).last_hidden_state[:, -1, :].detach()
                # num_neg, H
                negative_embeds = model.base_model.model(negative_state_ids).last_hidden_state[:, -1, :].detach()
        else:  # query turn
            slot_label = turn["slot_label"].to(device) if turn["slot_label"] is not None else None
            e_target = None
            negative_embeds = None

        # 前向传播
        with autocast(device_type="cuda", dtype=torch.bfloat16):
            loss_dict, m_new, new_len, g_logits = model(
                input_ids=input_ids,
                labels=labels,
                m_prev=m_prev,
                current_len=current_len,
                update_flag=update_flag,
                target_embed=e_target,
                negative_embeds=negative_embeds,
                slot_label=slot_label,
                trun_id=turn_id,
            )

        # 加权总 loss（可根据实验调整系数）
        loss = (
            loss_dict["llm_loss"]      * 1.0 +
            loss_dict["target_loss"]   * 10.0 +
            loss_dict["select_loss"]   * 3.0 +
            loss_dict["probe_loss"]    * 1.0
        )
        total_loss += loss
        loss_records.append({k: v.item() for k, v in loss_dict.items()})

        # 更新记忆状态（detach 防止梯度跨 turn 爆炸）
        m_prev = m_new.detach()
        current_len = new_len.detach()

        # ============ 指标收集（仅 update turn） ============
        if update_flag.item() > 0.5:
            with torch.no_grad():
                # 最新状态与权威 target 的余弦相似度
                sim = F.cosine_similarity(
                    F.normalize(m_new[:, -1, :], dim=-1),
                    F.normalize(e_target, dim=-1),
                    dim=-1
                ).cpu().item()
                sims_active.append(sim)

                g_prob = torch.sigmoid(g_logits).cpu().item()
                gates_pred_active.append(g_prob)
                # 真实 gate（teacher forcing 用的是 1.0）
                gates_active.append(1.0)  # 因为 update turn 强制更新

    # =============== 反向传播 ===============
    avg_loss = total_loss / len(batch["dialogue_turns"])
    avg_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad()

    # =============== 汇总 losses ===============
    losses = {k: sum(d[k] for d in loss_records) / len(loss_records) for k in loss_records[0]}
    losses["total_loss"] = avg_loss.item()

    # =============== 汇总 metrics ===============
    metrics = {
        "num_turns": len(batch["dialogue_turns"]),
        "num_updates": len(sims_active),
        "avg_sim_active": sum(sims_active) / len(sims_active) if sims_active else 0.0,
        "avg_gate_pred_active": sum(gates_pred_active) / len(gates_pred_active) if gates_pred_active else 0.0,
        "current_len_final": current_len.item(),
    }
    return losses, metrics


def train():
    # 1. 配置参数
    lr = 5e-5 # 建议初始微调可以稍微大一点，因为基座冻结了
    model_id = "/backup/lanzhenzhongLab/public/models/Qwen2.5-7B-Instruct"
    data_path = "mocker/mock_dialogues_multi_domain_istrain_1.json"
    save_path = "./checkpoints/cass_kv_qwen25_v1"
    os.makedirs(save_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在加载模型至: {device}")

    # 2. 模型与数据准备
    model = CASSKVInjectionModel(model_id).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, truncation_side="left")
    print("param dtype:", next(model.parameters()).dtype)

    # 假设你的 dataset 是一个 list 或 IterableDataset
    train_ds = load_dataset("json", data_files=data_path, split="train")
    collator = CASSKVSequentialCollator(tokenizer, max_length=1024)

    # 注意：由于 CASS 需要对话内时序，BatchSize 必须为 1 (一次处理一整个对话)
    train_loader = DataLoader(train_ds, batch_size=1, collate_fn=collator, shuffle=True)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # 初始化 WandB (可选)
    # wandb.init(project="CASS-KV-Project", name="Qwen2.5-7B")

    model.train()

    # 3. 训练循环
    epochs = 2
    global_step = 0

    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            losses, metrics = train_one_dialogue(model, batch, optimizer)
            loss_val = losses["total_loss"]
            target_loss = losses["target_loss"]
            select_loss = losses["select_loss"]
            pbar.set_postfix(
                {
                    "loss": f"{loss_val:.4f}",
                    "target": f"{target_loss:.4f}",
                    "select": f"{select_loss:.4f}",
                }
            )
            if global_step % 10 == 0:
                log_data = {
                    **{f"train/{k}": v for k, v in losses.items()},
                    **{f"train/{k}": v for k, v in metrics.items()}
                }
                # wandb.log(log_data, step=global_step)
                for k, v in log_data.items():
                    writer.add_scalar(k, v, global_step)

            if global_step % 100 == 0:
                save_checkpoint(model, optimizer, save_path, global_step)

            global_step += 1


def main():
    try:
        train()
    finally:
        writer.close()


if __name__ == "__main__":
    main()