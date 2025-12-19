import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

from model.kv import CASSKVInjectionModel


def update_kv_cache_with_memory(past_key_values, new_virtual_pkv):
    """
    past_key_values: 这里的格式是 tuple(tuple(layer_k, layer_v))
    new_virtual_pkv: 你通过新 m_t 计算出的 [(k, v), (k, v), ...]
    """
    new_past_key_values = []
    
    for layer_idx, (layer_k, layer_v) in enumerate(past_key_values):
        # layer_k 的形状通常是: (batch, num_heads, seq_len, head_dim)
        # 我们要替换的是 seq_len 维度中的第 0 个 index
        
        # 拿到这一层最新的注入 KV
        v_k, v_v = new_virtual_pkv[layer_idx] # (batch, num_heads, 1, head_dim)
        
        # 拼接：最新的注入 KV + 历史中除了第 0 位以外的所有 KV
        # 注意：这是最硬核的操作，确保历史 KV 的时空连续性
        updated_k = torch.cat([v_k, layer_k[:, :, 1:, :]], dim=2)
        updated_v = torch.cat([v_v, layer_v[:, :, 1:, :]], dim=2)
        
        new_past_key_values.append((updated_k, updated_v))
        
    return tuple(new_past_key_values)


class CASSPredictor:
    def __init__(self, model_id, adapter_path):
        print(f"正在加载基座模型: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # 1. 加载冻结的基座 (保持 BF16 精度)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CASSKVInjectionModel(model_id).to(self.device)
        
        # 2. 加载训练好的 CASS 权重
        print(f"正在注入 CASS 插件: {adapter_path}")
        if os.path.exists(adapter_path):
            state_dict = torch.load(adapter_path, map_location=self.device)
            # 这里的 state_dict 只包含 requires_grad=True 的部分
            self.model.load_state_dict(state_dict, strict=False)
        else:
            print("警告：未找到权重文件，将使用随机初始化状态进行测试。")
        
        self.model.eval()
        
        # 3. 初始化对话状态
        self.current_m = self.model.m_0
        self.history_ids = []
        self.current_pos = 1
        
        self.past_key_values = None

    @torch.no_grad()
    def interact(self, user_text):
        # 包装 Chat 模板 (不带 system message)
        messages = [{"role": "user", "content": user_text}]
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Tokenize
        new_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        input_ids = torch.tensor([new_ids]).to(self.device)
        
        # 构造位置 ID (从 current_pos 开始，给 M_t 留出位置 0)
        seq_len = input_ids.size(1)
        position_ids = torch.arange(
            self.current_pos, self.current_pos + seq_len, dtype=torch.long
        ).unsqueeze(0).to(self.device)

        # 1. 注入记忆 KV
        virtual_pkv = self.model.kv_projector(self.current_m)
        
        if self.past_key_values is not None:
            # 如果有缓存，把最前面的 M_t 换成最新的
            self.past_key_values = update_kv_cache_with_memory(self.past_key_values, virtual_pkv)
        else:
            # 第一轮，直接用 virtual_pkv 作为初始缓存
            self.past_key_values = virtual_pkv

        # 2. 调用基座生成回复
        # 我们使用 model.base_model 直接生成
        outputs = self.model.base_model.generate(
            input_ids=input_ids,
            past_key_values=self.past_key_values,
            use_cache=True,
            past_key_values=virtual_pkv,
            position_ids=position_ids,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id
        )

        # 提取回复文本
        response_ids = outputs[0][len(new_ids):]
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        # 3. 更新记忆状态 (模拟训练中的 Forward 逻辑)
        # 运行一次完整的 forward 拿到最后一个 token 的隐藏状态 r_agg
        full_out = self.model.base_model(
            input_ids=input_ids,
            past_key_values=virtual_pkv,
            position_ids=position_ids,
            output_hidden_states=False
        )
        r_agg = full_out.last_hidden_state[:, -1, :]
        
        # 计算门控信号 (Probe)
        g_logits = self.model.probe(torch.cat([r_agg, self.current_m], dim=-1))
        g_signal = torch.sigmoid(g_logits).item()
        
        # 合成新记忆 (Synthesis)
        m_uncontrolled = self.model.synthesis(r_agg, self.current_m)
        self.current_m = self.current_m + g_signal * (m_uncontrolled - self.current_m)

        # 更新位置偏移 (下一轮对话接着当前结束的位置)
        self.current_pos += (seq_len + len(response_ids))
        
        return response_text, g_signal

# --- 演示运行 ---
if __name__ == "__main__":
    # 指向你刚才保存的 checkpoint 路径
    ADAPTER_FILE = "./checkpoints/cass_qwen_v1/step_500/cass_adapter.pt"
    MODEL_ID = "/backup/lanzhenzhongLab/public/models/Qwen2.5-7B-Instruct"
    
    predictor = CASSPredictor(MODEL_ID, ADAPTER_FILE)
    
    print("\n--- CASS-KV 多领域对话系统已就绪 ---")
    while True:
        user_in = input("\nUser: ")
        if user_in.lower() in ['quit', 'exit', 'stop']: break
        
        res, g = predictor.interact(user_in)
        print(f"Update Signal (G): {g:.4f}")
        print(f"Assistant: {res}")