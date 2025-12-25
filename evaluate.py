from typing import Dict, Any
import os
from tqdm import tqdm
import pnlp
import torch
from torch.nn import functional as F
from torch.amp import autocast
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache

from utils import check_answer
from model.kv import CASSKVInjectionModel


class CASSEvaluator:
    def __init__(self, model: CASSKVInjectionModel, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()

    @torch.no_grad()
    def evaluate_item(self, item: Dict[str, Any]) -> str:
        TH = 0.5
        device = self.model.device
        # 初始化记忆队列
        m_prev = self.model.m_0.clone().to(device)          # (1, history_len, D)
        current_len = torch.tensor([1], device=device)      # 初始长度 1
        past_key_values = DynamicCache()  # KV cache 可复用
        dummy_labels = torch.zeros(1, 1, dtype=torch.long, device=device)

        # ==================== 遍历历史 turns，更新记忆 ====================
        for turn in item["dialogue_turns"]:
            messages = [{"role": "user", "content": turn["user_input"]}]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            input_ids = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(device)

            loss_dict, m_prev, current_len, g_logits = self.model(
                input_ids=input_ids,
                labels=dummy_labels,
                m_prev=m_prev,
                current_len=current_len,
                update_flag=torch.tensor(1.0, device=device),  # 更新 memory
            )
            g_prob = torch.sigmoid(g_logits)
            g_signal = (g_prob > TH).float()  # (1, 1)
            # print(f"g_signal: {g_signal}")

        # ==================== 处理最终 test query ====================
        test_query = item.get("test_query", "当前状态是什么？")
        messages = [{"role": "user", "content": test_query}]
        query_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        query_ids = self.tokenizer.encode(query_prompt, return_tensors="pt").to(device)

        # 获取 query 的 r_agg，用于历史位置选择
        token_embeds = self.model.base_model.get_input_embeddings()(query_ids)
        outputs = self.model.base_model.model(inputs_embeds=token_embeds, use_cache=False)
        r_agg = outputs.last_hidden_state[:, -1, :]

        # ---- slot selection（hard top-1）----
        positions = torch.arange(self.model.history_len, device=device)
        time_bias = self.model.time_emb(positions).unsqueeze(0)
        m_read = (
            m_prev.detach() 
            # + 0.5 * self.model.read_content_proj(m_prev.detach())
        )

        slot_logits = self.model.slot_selector(r_agg.unsqueeze(1), m_read, time_bias)  # (1, history_len)
        mask = torch.arange(self.model.history_len, device=device).unsqueeze(0) < current_len
        slot_logits = slot_logits.masked_fill(~mask, -1e9)
        slot_probs = F.softmax(slot_logits, dim=-1)
        selected_idx = slot_logits.argmax(dim=-1, keepdim=True)
        selected_m = m_prev.gather(
            dim=1,
            index=selected_idx.unsqueeze(-1).expand(-1, -1, m_prev.size(-1))
        )

        latest_kv = self.model.latest_projector(m_prev[:, -1:, :])
        history_kv = self.model.history_projector(selected_m)

        # 投影 KV
        latest_kv = self.model.latest_projector(m_prev[:, -1:, :])      # length=1
        history_kv = self.model.history_projector(selected_m)           # length=1

        # 合并：最新 + 选中历史（共 2 个虚拟 token）
        past_key_values = DynamicCache()
        for layer_idx in range(self.model.config.num_hidden_layers):
            k_latest, v_latest = latest_kv[layer_idx]
            k_hist, v_hist = history_kv[layer_idx]
            k_comb = torch.cat([k_latest, k_hist], dim=2)               # (1, heads, 2, head_dim)
            v_comb = torch.cat([v_latest, v_hist], dim=2)
            past_key_values.update(k_comb, v_comb, layer_idx)

        # ================= 4. Generate =================
        prefix_len = 2
        position_ids = torch.arange(prefix_len, prefix_len + query_ids.size(1), device=device).unsqueeze(0)

        generated = self.model.base_model.generate(
            input_ids=query_ids,
            past_key_values=past_key_values,
            position_ids=position_ids,
            max_new_tokens=64,
            do_sample=False,
            temperature=0.2,
            repetition_penalty=1.5,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
        )
        output_text = self.tokenizer.decode(
            generated[0][query_ids.size(1):],
            skip_special_tokens=True
        )
        return {
            "prediction": output_text,
            "probe_prob": g_prob,
            "slot_prob": slot_probs,
            "selected_idx": selected_idx.item(),
        }


async def main():
    model_id = "/backup/lanzhenzhongLab/public/models/Qwen2.5-7B-Instruct"
    adapter_path = "./checkpoints/cass_kv_qwen25_v1/step_1800/cass_adapter.pt"
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
    evaluator = CASSEvaluator(model, tokenizer)

    file = "data_simple/eval_ds_human.jsonl"
    file = "mocker/mock_dialogues_multi_domain_istrain_0.json"
    if file.endswith("json"):
        ds = pnlp.read_json(file)
    else:
        ds = pnlp.read_file_to_list_dict(file)

    results = []
    correct = 0
    for item in tqdm(ds):
        gt = item["ground_truth"]
        slot_label = item["slot_label"]
        test_query = item["test_query"]
        response = evaluator.evaluate_item(item)
        pred = response["prediction"]
        slot_label_pred= response["selected_idx"]
        slot_prob = response["slot_prob"]

        print("\n\n" + "=" * 80)
        print(f"Query: {test_query}")
        print(f"GroundTruth: slot_label={slot_label}, answer={gt!r}")
        print(f"Prediction:  slot_label={slot_label_pred}, answer={pred!r}")
        print(f"slot_prob: {slot_prob}")
        print("=" * 80)


        is_correct = False # await check_answer(pred, gt)
        res = {
            "is_correct": is_correct,
            "response": pred,
            "ground_truth": gt,
            "thinking": "",
        }
        results.append(res)
        if is_correct:
            correct += 1

    accuracy = correct / len(results) if results else 0
    print(f"Accuracy: {accuracy:.4f} ({correct}/{len(ds)})")

    out_file = file.rsplit(".jsonl", 1)[0] + f"_eval_{model_tag}.jsonl"
    pnlp.write_list_dict_to_file(out_file, results)
    print(f"Saved to {out_file}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())