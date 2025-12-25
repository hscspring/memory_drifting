from typing import List, Dict, Any
import os
import re

from tqdm import tqdm
import pnlp
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache


from utils import check_answer
from model.kv import CASSKVInjectionModel, get_state_embedding



class CASSEvaluator:
    def __init__(self, model: CASSKVInjectionModel, tokenizer, use_ground_state: bool = False):
        self.model = model
        self.tokenizer = tokenizer
        self.use_ground_state = use_ground_state
        self.model.eval()

    @torch.no_grad()
    def evaluate_item(self, item: Dict[str, Any]) -> str:
        device = self.model.device
        self.model.reset_memory()

        # 初始化记忆队列
        m_prev = self.model.m_0.clone().to(device)  # (1, history_len, D)
        current_len = torch.tensor([1], device=device)  # 初始只有最新槽位（空）

        # 遍历所有历史 turn，逐步更新记忆
        for turn in item["dialogue_turns"]:
            user_input = turn["user_input"]
            messages = [{"role": "user", "content": user_input}]
            if turn.get("update_flag", 0) == 0:
                # query turn 需要带 assistant 输出用于完整上下文
                messages.append({"role": "assistant", "content": turn["llm_output"]})

            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(device)

            update_flag = torch.tensor([1.0 if turn.get("update_flag", 0) == 1 else 0.0], device=device)

            # 获取 r_agg（不注入任何记忆，仅用于更新或检索）
            token_embeds = self.model.base_model.get_input_embeddings()(input_ids)
            outputs = self.model.base_model.model(inputs_embeds=token_embeds, use_cache=False)
            r_agg = outputs.last_hidden_state[:, -1, :]

            # 更新记忆（与训练完全一致）
            latest_m = m_prev[:, -1, :]
            g_logits = self.model.probe(torch.cat([r_agg, latest_m], dim=-1))
            g_signal = (torch.sigmoid(g_logits) > 0.5).float()  # 推理时用预测的 gate

            if update_flag.item() > 0.5 and turn.get("authoritative_state_M_t") and self.use_ground_state:
                # 如果有权威状态文本（仅用于调试或半监督），强制写入
                state_emb = get_state_embedding(self.model.base_model, self.tokenizer, turn["authoritative_state_M_t"], device)
                if state_emb is not None:
                    m_candidate = state_emb
                else:
                    m_candidate = self.model.synthesis(torch.cat([r_agg, latest_m], dim=-1))
            else:
                m_candidate = self.model.synthesis(torch.cat([r_agg, latest_m], dim=-1))

            m_new_latest = g_signal * m_candidate + (1 - g_signal) * latest_m
            m_new = torch.roll(m_prev, shifts=-1, dims=1)
            m_new[:, -1, :] = m_new_latest

            if update_flag.item() > 0.5:
                current_len = torch.min(current_len + 1, torch.tensor(self.model.history_len, device=device))

            m_prev = m_new

        # ==================== 处理最终 test query ====================
        test_query = item.get("test_query", "当前状态是什么？")
        messages = [{"role": "user", "content": test_query}]
        query_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        query_ids = self.tokenizer.encode(query_prompt, return_tensors="pt").to(device)

        token_embeds = self.model.base_model.get_input_embeddings()(query_ids)
        outputs = self.model.base_model.model(inputs_embeds=token_embeds, use_cache=False)
        r_agg = outputs.last_hidden_state[:, -1, :]

        # 历史位置选择
        mask = torch.arange(self.model.history_len, device=device).unsqueeze(0) < current_len
        slot_logits = self.model.slot_selector(r_agg)
        slot_logits = slot_logits.masked_fill(~mask, -1e9)
        slot_probs = F.softmax(slot_logits, dim=-1)

        
        # top_k = 3  # 可调 2~5
        # topk_indices = slot_probs.topk(top_k, dim=-1).indices  # (1, top_k)，值是位置索引，如 [7, 6, 8]
        # selected_m = m_prev.gather(
        #     dim=1,
        #     index=topk_indices.unsqueeze(-1).expand(-1, top_k, m_prev.size(-1))  # (1, top_k, hidden_size)
        # )

        # topk=1
        selected_idx = slot_probs.argmax(dim=-1, keepdim=True)  # (1,1)
        selected_idx = selected_idx.clamp(max=current_len-1)   # 安全
        selected_m = m_prev.gather(1, selected_idx.unsqueeze(-1).expand(-1, -1, m_prev.size(-1)))
        
        # 原始版本
        # selected_m = torch.sum(m_prev * slot_probs.unsqueeze(-1), dim=1, keepdim=True)

        latest_kv = self.model.latest_projector(m_prev[:, -1:, :])
        history_kv = self.model.history_projector(selected_m)

        print("Slot probs top5:", slot_probs.topk(5, dim=-1))
        print("Current len:", current_len.item())
        print("Probs sum:", slot_probs.sum().item())

        past_key_values = DynamicCache()
        for layer_idx in range(self.model.config.num_hidden_layers):
            k_latest, v_latest = latest_kv[layer_idx]
            k_hist, v_hist = history_kv[layer_idx]
            k_comb = torch.cat([k_latest, k_hist], dim=2)
            v_comb = torch.cat([v_latest, v_hist], dim=2)
            past_key_values.update(k_comb, v_comb, layer_idx)

        # 生成
        input_len = query_ids.size(1)  # query 的 token 数
        prefix_len = 2
        position_ids = torch.arange(prefix_len, prefix_len + input_len, device=device).unsqueeze(0)
        generated = self.model.base_model.generate(
            # inputs_embeds=token_embeds,
            input_ids=query_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            max_new_tokens=128,
            do_sample=False,  # 评估时确定性
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
            repetition_penalty=1.2,
        )

        prediction = self.tokenizer.decode(generated[0][len(query_ids[0]):], skip_special_tokens=True)
        return prediction


async def main():
    model_id = "/backup/lanzhenzhongLab/public/models/Qwen2.5-7B-Instruct"
    adapter_path = "./checkpoints/cass_kv_qwen25_v1/step_800/cass_adapter.pt"
    model_tag = "qwen25_7b_ft"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CASSKVInjectionModel(model_id)
    print(f"正在注入 CASS 插件: {adapter_path}")
    if os.path.exists(adapter_path):
        state_dict = torch.load(adapter_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print("注入成功！")
    else:
        print("警告：未找到权重文件，将使用随机初始化状态进行测试。")
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id, truncation_side="left")
    evaluator = CASSEvaluator(model, tokenizer, False)

    file = "data_simple/eval_ds_human.jsonl"
    ds = pnlp.read_file_to_list_dict(file)
    results = []
    correct = 0
    for item in tqdm(ds):
        gt = item.get("ground_truth", "")
        # try:
        response = evaluator.evaluate_item(item)
        print("\n\n" + "=" * 80)
        print(f"GT: {gt!r}")
        print(f"Response: {response!r}")
        print("=" * 80)
        is_correct = await check_answer(response, gt)  # 假设check_answer是async
        res = {
            "is_correct": is_correct,
            "response": response,
            "ground_truth": gt,
            "thinking": "",
        }
        results.append(res)
        if is_correct:
            correct += 1
        # except Exception as e:
        #     print(f"Error on item: {e}")
        #     continue  # 跳过错误item

    n_correct = correct  # 用累加代替sum
    accuracy = n_correct / len(results) if results else 0
    print(f"Accuracy: {accuracy:.4f} ({n_correct}/{len(ds)})")  # 用len(ds)总计
    out_file = file.rsplit(".jsonl", 1)[0] + f"_eval_{model_tag}.jsonl"  # 更鲁棒replace
    pnlp.write_list_dict_to_file(out_file, results)
    print(f"Saved to {out_file}")

# 运行async main
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())