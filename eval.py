import torch
import json
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache



class CASSEvaluator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device

    @torch.no_grad()
    def evaluate_item(self, item):
        """处理单条评估数据"""
        m_prev = self.model.m_0
        current_pos = 1
        past_key_values = None
        
        # 1. 对话流转：建立历史 KV Cache 并更新记忆
        for turn in item['dialogue_turns']:
            # 包装输入
            msgs = [
                {"role": "user", "content": turn['user_input']},
                {"role": "assistant", "content": turn['llm_output']}
            ]
            prompt = self.tokenizer.apply_chat_template(
                msgs,
                tokenize=False
            )
            input_ids = torch.tensor([self.tokenizer.encode(prompt, add_special_tokens=False)]).to(self.device)
            seq_len = input_ids.size(1)
            
            # 注入当前记忆并推理（提取下一个 M_t 需要的 r_agg）
            # 注意：评估时，我们完全使用模型预测的 g 信号，不参考 update_flag
            virtual_pkv = DynamicCache()
            for layer_idx, (k, v) in enumerate(self.model.kv_projector(m_prev)):
                virtual_pkv.update(
                    key_states=k,
                    value_states=v,
                    layer_idx=layer_idx,
                )

            # 这里我们运行一次 forward 模拟处理历史
            outputs = self.model.base_model(
                input_ids=input_ids,
                past_key_values=virtual_pkv,
                position_ids=torch.arange(current_pos, current_pos + seq_len).unsqueeze(0).to(self.device),
                output_hidden_states=False
            )
            
            # 更新记忆逻辑
            r_agg = outputs.last_hidden_state[:, -1, :]
            g_logits = self.model.probe(torch.cat([r_agg, m_prev], dim=-1))
            g_signal = torch.sigmoid(g_logits)
            m_uncontrolled = self.model.synthesis(r_agg, m_prev)
            m_prev = m_prev + g_signal * (m_uncontrolled - m_prev)
            
            # 更新位置（为了最后的 Query 能接在历史后面）
            current_pos += seq_len

        # 2. 最终测试：回答 Test Query
        query_prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": item['test_query']}],
            tokenize=False, add_generation_prompt=True
        )
        query_ids = torch.tensor([self.tokenizer.encode(query_prompt, add_special_tokens=False)]).to(self.device)
        
        # 使用最后一轮的 m_prev 注入
        final_pkv = DynamicCache()
        for layer_idx, (k, v) in enumerate(self.model.kv_projector(m_prev)):
            final_pkv.update(
                key_states=k,
                value_states=v,
                layer_idx=layer_idx,
            )
        
        # 我们可以选择是“全量重算历史”还是“增量推理”
        # 批量评估建议全量重算比较稳妥，因为我们要评估的是模型在长上下文下的索引
        # 这里为了简单，演示逻辑一致性：
        generation = self.model.base_model.generate(
            input_ids=query_ids,
            past_key_values=final_pkv, # 注入最权威的当前状态
            max_new_tokens=64,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        prediction = self.tokenizer.decode(generation[0], skip_special_tokens=True)
        # 简单清洗，只取 Assistant 的回复部分
        if "assistant" in prediction.lower():
            prediction = prediction.split("assistant")[-1].strip()
            
        return prediction

def main():
    # 加载模型和权重 (同 Inference.py 逻辑)
    # ... 
    evaluator = CASSEvaluator(model, tokenizer)
    
    with open("test_data.json", "r") as f:
        test_set = json.load(f)
    
    results = []
    correct = 0
    
    for item in tqdm(test_set):
        pred = evaluator.evaluate_item(item)
        gt = item['ground_truth']
        
        # 简单的字符串匹配评估 (实际可以使用 ROUGE 或 LLM-as-a-judge)
        is_correct = gt in pred
        if is_correct: correct += 1
        
        results.append({
            "query": item['test_query'],
            "pred": pred,
            "gt": gt,
            "match": is_correct
        })
    
    print(f"Accuracy: {correct / len(test_set):.2%}")
    # 保存结果方便 Debug
    with open("eval_results.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)