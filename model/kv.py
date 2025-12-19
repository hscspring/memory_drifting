import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache


def compute_target_loss(m_new, target_embed, negatives=None, alpha=0.5, tau=0.05):
    """
    m_new: (B, hidden_size)
    target_embed: (B, hidden_size)
    negatives: list of tensors [(B, hidden_size), ...] 可选
    alpha: 权重混合 MSE 和 contrastive
    tau: contrastive 温度
    """
    # 归一化向量
    m_norm = F.normalize(m_new, dim=-1)
    t_norm = F.normalize(target_embed, dim=-1)

    return 1 - F.cosine_similarity(m_norm, t_norm, dim=-1)

    # 1. 余弦相似度 MSE
    mse_loss = F.mse_loss(m_norm, t_norm)

    # 2. contrastive loss
    if negatives is not None:
        pos_sim = (m_norm * t_norm).sum(-1) / tau  # (B,)
        neg_sims = torch.stack([(m_norm * n).sum(-1) / tau for n in negatives], dim=0)  # (num_neg, B)
        logits = torch.cat([pos_sim.unsqueeze(0), neg_sims], dim=0)  # (1 + num_neg, B)
        labels = torch.zeros(m_norm.size(0), dtype=torch.long, device=m_new.device)  # 正样本 index=0
        contrastive_loss = F.cross_entropy(logits.T, labels)  # batch-wise
        # 混合 loss
        return alpha * mse_loss + (1 - alpha) * contrastive_loss
    else:
        return mse_loss


class KVProjector(nn.Module):
    def __init__(self, config, use_layernorm=True):
        super().__init__()
        self.num_layers = config.num_hidden_layers
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.use_layernorm = use_layernorm

        self.projections = nn.ModuleList()
        self.gamma_layers = nn.ParameterList()  # layer-wise scale

        for _ in range(self.num_layers):
            mlp = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, 2 * self.num_kv_heads * self.head_dim)
            )
            self.projections.append(mlp)
            self.gamma_layers.append(nn.Parameter(torch.ones(1)))  # 初始化为1.0

        if self.use_layernorm:
            self.ln = nn.LayerNorm(self.hidden_size)

        # 初始化 Linear 层权重
        for m in self.projections:
            for layer in m:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0.0)

    def forward(self, m_t):
        """
        m_t: (B, hidden_size)
        返回: tuple of (k, v) per layer
        """
        if self.use_layernorm:
            m_t = self.ln(m_t)

        batch_size = m_t.size(0)
        past_key_values = []

        for i, mlp in enumerate(self.projections):
            kv = mlp(m_t) * self.gamma_layers[i]  # layer-wise scale
            kv = kv.view(batch_size, 2, self.num_kv_heads, 1, self.head_dim)
            k = kv[:, 0]  # (B, num_heads, 1, head_dim)
            v = kv[:, 1]  # (B, num_heads, 1, head_dim)
            past_key_values.append((k, v))

        return tuple(past_key_values)


class CASSKVInjectionModel(nn.Module):
    def __init__(self, model_id):
        super().__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        self.config = self.base_model.config
        self.hidden_size = self.config.hidden_size

        dtype = self.base_model.dtype
        self.read_emb = nn.Parameter(
            torch.randn(1, 1, self.hidden_size, dtype=dtype)
        )
        self.m_0 = nn.Parameter(
            torch.randn(1, self.hidden_size, dtype=dtype)
        )
        self.kv_projector = KVProjector(self.config, use_layernorm=True).to(dtype=dtype)
        self.probe = nn.Linear(2 * self.hidden_size, 1, dtype=dtype)
        self.synthesis = nn.GRUCell(self.hidden_size, self.hidden_size).to(dtype=dtype)
        nn.init.normal_(self.read_emb, std=0.02)
        nn.init.normal_(self.m_0, std=1)
    
    @property
    def device(self):
        return self.m_0.device

    def forward(self, input_ids, labels, m_prev, update_flag, target_embed, negative_embed):
        # negative_embed: B,H
        device = input_ids.device
        m_prev = m_prev.to(self.base_model.dtype)
        update_flag = update_flag.to(self.base_model.dtype)

        token_embeds = self.base_model.get_input_embeddings()(input_ids)
        read_emb = self.read_emb.expand(
            token_embeds.size(0), 1, -1
        )
        inputs_embeds = torch.cat(
            [read_emb, token_embeds],
            dim=1
        )

        labels = torch.cat(
            [
                torch.full(
                    (labels.size(0), 1),
                    -100,
                    device=device,
                    dtype=labels.dtype
                ),
                labels
            ],
            dim=1
        )


        virtual_pkv = DynamicCache()
        kv_pairs = self.kv_projector(m_prev.detach())

        for layer_idx, (k, v) in enumerate(kv_pairs):
            virtual_pkv.update(
                key_states=k,
                value_states=v,
                layer_idx=layer_idx,
            )
        outputs = self.base_model.model(
            inputs_embeds=inputs_embeds,
            past_key_values=virtual_pkv,
            use_cache=False,
        )
        hidden_states = outputs.last_hidden_state
        slice_indices = slice(0, None)
        logits = self.base_model.lm_head(hidden_states[:, slice_indices, :])
        lm_loss = self.base_model.loss_function(
            logits=logits, labels=labels, vocab_size=self.config.vocab_size
        )

        r_agg = outputs.last_hidden_state[:, -1, :]
        # _r_mem = outputs.last_hidden_state[:, 0, :]

        g_logits = self.probe(torch.cat([r_agg, m_prev], dim=-1))
        probe_loss = F.binary_cross_entropy_with_logits(g_logits, update_flag.view(-1, 1))

        m_uncontrolled = self.synthesis(r_agg, m_prev)
        # 训练阶段建议使用真实的 update_flag 进行记忆流转 (Teacher Forcing)，这样即使 probe 预测错了，M 也能学到正确的语义
        g_for_update = update_flag.view(-1, 1) 
        m_new = m_prev + g_for_update * (m_uncontrolled - m_prev)
        target_loss = compute_target_loss(m_new, target_embed, negatives=negative_embed, alpha=0.5)
        loss_dict = {
            "llm_loss": lm_loss,
            "target_loss": target_loss,
            "probe_loss": probe_loss,
        }
        return loss_dict, m_new, g_logits



if __name__ == "__main__":
    import torch
    model_id = "/backup/lanzhenzhongLab/public/models/Qwen2.5-7B-Instruct"
    model = CASSKVInjectionModel(model_id)
    x = torch.rand(1, 3584)
    res = model.kv_projector(x)
    print([(v[0].shape,v[1].shape) for v in res])
