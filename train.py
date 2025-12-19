import os
from datetime import datetime

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from transformers import AutoTokenizer
from tqdm import tqdm
import wandb
from datasets import load_dataset
from torch.utils.tensorboard import SummaryWriter


from model.kv import CASSKVInjectionModel
from dataset.kv import CASSKVSequentialCollator
from model_utils import save_checkpoint



log_dir = os.path.join(
    "tensorboard_log/",
    datetime.now().strftime("%Y%m%d-%H%M%S")
)

writer = SummaryWriter(log_dir=log_dir)


def train_one_dialogue(model, batch, optimizer):
    m_prev = model.m_0 # 初始记忆向量
    total_dialogue_loss = 0
    sims_active = []
    sims_passive = []
    gates_active = []
    gates_passive = []
    loss_lst = []

    # model.base_model.gradient_checkpointing_enable()
    for idx, turn in enumerate(batch["dialogue_turns"]):
        input_ids = turn["input_ids"].unsqueeze(0).to(model.device)
        labels = turn["labels"].unsqueeze(0).to(model.device)
        target_ids = turn["target_ids"].unsqueeze(0).to(model.device)
        negative_state_ids = turn["negative_state_ids"].to(model.device)
        update_flag = turn["update_flag"].to(model.device)

        # 1. 提取语义目标 E_target
        with torch.no_grad():
            target_outputs = model.base_model.model(target_ids)
            e_target = target_outputs.last_hidden_state[:, -1, :].detach()
            negative_outputs = model.base_model.model(negative_state_ids)
            e_negative = negative_outputs.last_hidden_state[:, -1, :].detach()

        # 2. 前向传播
        with autocast(device_type="cuda", dtype=torch.bfloat16):
            loss_dict, m_new, g_logits = model(
                input_ids, labels, m_prev, 
                update_flag, e_target, e_negative
            )
            loss_lst.append(loss_dict)
            llm_loss = loss_dict["llm_loss"]
            target_loss = loss_dict["target_loss"]
            probe_loss = loss_dict["probe_loss"]
            loss = llm_loss + 1.0 * target_loss + 0.01 * probe_loss
            total_dialogue_loss += loss
        # torch.cuda.reset_peak_memory_stats()
        # print("max mem MB:", torch.cuda.max_memory_allocated() / 1024**2)

        with torch.no_grad():
            sim = F.cosine_similarity(
                F.normalize(m_new, dim=-1),
                F.normalize(e_target, dim=-1),
                dim=-1
            ).item()
            g_val = torch.sigmoid(g_logits).item()
            if turn["update_flag"] > 0.5:
                sims_active.append(sim)
                gates_active.append(g_val)
            else:
                sims_passive.append(sim)
                gates_passive.append(g_val)

        # 3. 记忆流转 (detach 防止跨轮梯度爆炸)
        m_prev = m_new.detach()

    # 4. 反向传播与更新
    # 均摊到每一轮的平均 loss
    avg_loss = total_dialogue_loss / len(batch["dialogue_turns"])
    avg_loss.backward()

    # 梯度裁剪，防止 Transformer 训练中常见的梯度爆炸
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    optimizer.zero_grad()

    metrics = {
        "avg_sim_active": sum(sims_active)/len(sims_active) if sims_active else 0,
        "avg_sim_passive": sum(sims_passive)/len(sims_passive) if sims_passive else 0,
        "avg_gate_active": sum(gates_active) / len(gates_active) if gates_active else 0,
        "avg_gate_passive": sum(gates_passive) / len(gates_passive) if gates_passive else 0,
    }
    metrics["num_active_updates"] = len(sims_active)
    metrics["num_passive_updates"] = len(sims_passive)
    
    losses = {
        "loss": avg_loss.item(),
        "llm_loss": sum([v["llm_loss"] for v in loss_lst])/len(loss_lst) if loss_lst else 0,
        "target_loss": sum([v["target_loss"] for v in loss_lst])/len(loss_lst) if loss_lst else 0,
        "probe_loss": sum([v["probe_loss"] for v in loss_lst])/len(loss_lst) if loss_lst else 0,
    }
    return losses, metrics


def train():
    # 1. 配置参数
    lr = 5e-5 # 建议初始微调可以稍微大一点，因为基座冻结了
    model_id = "/backup/lanzhenzhongLab/public/models/Qwen2.5-7B-Instruct"
    data_path = "mocker/kv_mock_singleslot_2000.jsonl"
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
    epochs = 1
    global_step = 0

    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            losses, metrics = train_one_dialogue(model, batch, optimizer)
            loss_val = losses["loss"]
            pbar.set_postfix({"loss": f"{loss_val:.4f}"})
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