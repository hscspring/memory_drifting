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

    # return 1 - F.cosine_similarity(m_norm, t_norm, dim=-1)

    # 1. 余弦相似度 MSE
    mse_loss = F.mse_loss(m_norm, t_norm)

    # 2. contrastive loss
    if negatives is not None:
        pos_sim = (m_norm * t_norm).sum(-1) / tau  # (B,)
        neg_norms = [F.normalize(n, dim=-1) for n in negatives]
        neg_sims = torch.stack(
            [(m_norm * n).sum(-1) / tau for n in neg_norms],
            dim=0
        )
        logits = torch.cat([pos_sim.unsqueeze(0), neg_sims], dim=0)  # (1 + num_neg, B)
        labels = torch.zeros(m_norm.size(0), dtype=torch.long, device=m_new.device)  # 正样本 index=0
        contrastive_loss = F.cross_entropy(logits.T, labels)  # batch-wise
        # 混合 loss
        return alpha * mse_loss + (1 - alpha) * contrastive_loss
    else:
        return mse_loss


class CASSLightModel(nn.Module):
    def __init__(self, model_id, prefix_len=4):
        super().__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.config = self.base_model.config
        self.hidden_size = self.config.hidden_size
        self.prefix_len = prefix_len  # 注入的虚拟 token 数量，推荐 4~8

        dtype = self.base_model.dtype

        # 1. 单个外部状态向量 M_t（最新状态摘要）
        self.m_0 = nn.Parameter(torch.randn(1, self.hidden_size, dtype=dtype))
        nn.init.normal_(self.m_0, std=0.02)

        # 2. 可选：一个全局 read token（许多工作证明有用）
        self.read_emb = nn.Parameter(torch.randn(1, 1, self.hidden_size, dtype=dtype))
        nn.init.normal_(self.read_emb, std=0.02)

        # 3. KV Projector：从 M_t 生成 prefix_len 个虚拟 KV（每层独立）
        self.kv_projector = nn.ModuleList()
        for _ in range(self.config.num_hidden_layers):
            mlp = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, 2 * self.prefix_len * self.config.hidden_size)  # 输出 K 和 V 拼一起
            )
            self.kv_projector.append(mlp)

        # 4. Synthesis GRU：更新 M_t
        self.synthesis = nn.GRUCell(self.hidden_size, self.hidden_size)

        # 5. HSP 探测器：判断是否需要更新状态
        self.probe = nn.Linear(self.hidden_size, 1)  # 从 r_agg 预测 update prob

        # 6. 新增：相对偏移预测头（处理历史查询）
        #    方案A：回归实数偏移（简单）
        self.offset_head = nn.Linear(self.hidden_size, 1)
        #    方案B：分类最近 K 个（更稳健，推荐 K=10）
        # self.offset_head = nn.Linear(self.hidden_size, 11)  # -10 到 0（0=最新）

    def get_prefix_kv(self, m_t):
        """
        m_t: (B, hidden_size)
        返回: list of (k, v) per layer, each k/v: (B, num_kv_heads, prefix_len, head_dim)
        """
        B = m_t.size(0)
        past_key_values = []
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        num_kv_heads = self.config.num_key_value_heads

        for layer_idx, proj in enumerate(self.kv_projector):
            kv_flat = proj(m_t)  # (B, 2 * prefix_len * hidden_size)
            kv_flat = kv_flat.view(B, self.prefix_len, 2, self.hidden_size)
            k = kv_flat[:, :, 0].view(B, self.prefix_len, num_kv_heads, head_dim).transpose(1, 2)
            v = kv_flat[:, :, 1].view(B, self.prefix_len, num_kv_heads, head_dim).transpose(1, 2)
            past_key_values.append((k, v))

        return tuple(past_key_values)

    def forward(
        self,
        input_ids,
        labels,
        m_prev,              # (1, hidden_size)
        update_flag,         # 1.0 更新 / 0.0 查询
        target_embed=None,   # 更新时：最新状态的 frozen embedding
        negative_embed=None,
        offset_label=None,   # 查询时：相对偏移标签，如 tensor([-1.0]) 表示“上一次”
    ):
        device = input_ids.device
        B = input_ids.size(0)
        m_prev = m_prev.to(self.base_model.dtype)

        # ====== 1. 输入 embedding + 可选 read token ======
        token_embeds = self.base_model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([self.read_emb.expand(B, -1, -1), token_embeds], dim=1)

        # labels 相应 pad 一列 -100
        pad_labels = torch.full((B, 1), -100, device=device, dtype=labels.dtype)
        labels = torch.cat([pad_labels, labels], dim=1)

        # ====== 2. 生成 prefix KV（从最新 M_t）======
        prefix_kv = self.get_prefix_kv(m_prev.unsqueeze(0))  # add batch dim

        # ====== 3. 一次完整 forward，得到 r_agg（最后一轮的隐藏状态）======
        outputs = self.base_model(
            inputs_embeds=inputs_embeds,
            past_key_values=prefix_kv,
            use_cache=False,
        )
        hidden_states = outputs.last_hidden_state
        r_agg = hidden_states[:, -1, :]  # (B, hidden)

        # ====== 4. 计算各损失 ======
        lm_loss = 0.0
        if update_flag.item() < 0.5:  # query turn
            logits = self.base_model.lm_head(hidden_states)
            lm_loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )

        # Probe loss：预测是否更新
        probe_logit = self.probe(r_agg)
        probe_loss = F.binary_cross_entropy_with_logits(probe_logit.squeeze(-1), update_flag)

        # Memory update（teacher forcing 用真实 flag）
        g = update_flag.unsqueeze(-1)  # (B,1)
        m_candidate = self.synthesis(r_agg, m_prev)
        m_new = g * m_candidate + (1 - g) * m_prev

        # Target alignment loss（更新轮）
        target_loss = 0.0
        if update_flag.item() > 0.5 and target_embed is not None:
            target_loss = compute_target_loss(m_new, target_embed, negative_embed)

        # 新增：相对偏移预测 loss（查询轮）
        offset_loss = 0.0
        if offset_label is not None:
            pred_offset = self.offset_head(r_agg).squeeze(-1)  # (B,)
            offset_loss = F.mse_loss(pred_offset, offset_label)  # 或 cross_entropy 如果分类

        loss_dict = {
            "lm_loss": lm_loss,
            "target_loss": target_loss,
            "probe_loss": probe_loss,
            "offset_loss": offset_loss,
        }

        return loss_dict, m_new.squeeze(0)  # 返回去掉 batch dim 的 m_new 用于下一轮