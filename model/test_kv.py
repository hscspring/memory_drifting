import torch
from transformers import AutoConfig

from kv import KVProjector


def verify_kv_projection(model_id):
    print(f"正在加载 {model_id} 的配置...")
    config = AutoConfig.from_pretrained(model_id)
    
    # 2. 检查关键 GQA 参数
    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // config.num_attention_heads
    hidden_size = config.hidden_size
    
    print(f"模型层数: {num_layers}")
    print(f"KV 头数 (GQA): {num_kv_heads}")
    print(f"每个 Head 的维度: {head_dim}")
    print("-" * 30)

    # 3. 实例化你的投影器
    projector = KVProjector(config) # 使用你刚才的代码逻辑
    
    # 模拟一个 Batch 的 M_t
    batch_size = 2
    m_t = torch.randn(batch_size, hidden_size)
    
    # 4. 执行投影
    virtual_pkv = projector(m_t)
    
    # 5. 维度校验逻辑
    assert len(virtual_pkv) == num_layers, "层数不匹配！"
    
    for i, (k, v) in enumerate(virtual_pkv):
        # 预期的形状: (batch, num_kv_heads, seq_len=1, head_dim)
        expected_shape = (batch_size, num_kv_heads, 1, head_dim)
        
        if k.shape != expected_shape or v.shape != expected_shape:
            print(f"❌ 第 {i} 层形状错误！")
            print(f"   预期: {expected_shape}")
            print(f"   实际: K={tuple(k.shape)}, V={tuple(v.shape)}")
            return
    
    print("✅ 维度验证通过！所有注入的 KV 缓存与 Qwen2.5 GQA 架构完美对齐。")
    
    # 计算参数量
    total_params = sum(p.numel() for p in projector.parameters())
    print(f"📊 KVProjector 总参数量: {total_params / 1e6:.2f} M (仅占 7B 模型的约 0.1%)")

# 执行验证
# (确保 QwenKVProjector 类已在上方定义)
model_id = "/backup/lanzhenzhongLab/public/models/Qwen2.5-7B-Instruct"
verify_kv_projection(model_id)