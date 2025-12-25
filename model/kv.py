from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache, Cache



def compute_target_loss_contrastive(m_new, target_embed, negatives=None, alpha=0.5, tau=0.05):
    """
    m_new: (1, H)
    target_embed: (1, H)
    negatives: (num_neg, H)
    alpha: 权重混合 MSE 和 contrastive
    tau: contrastive 温度
    """
    # 归一化向量
    m_norm = F.normalize(m_new, dim=-1)
    t_norm = F.normalize(target_embed, dim=-1)

    # 1. 余弦相似度 MSE
    mse_loss = F.mse_loss(m_norm, t_norm) # scalar
    if negatives is None or len(negatives) == 0:
        return mse_loss

    # 2. contrastive loss
    pos_sim = (m_norm * t_norm).sum(-1) / tau  # (1,)
    neg_norms = [F.normalize(n, dim=-1) for n in negatives]
    neg_sims = torch.stack(
        [(m_norm * n).sum(-1) / tau for n in neg_norms],
        dim=0
    ) # (num_neg, 1)
    logits = torch.cat([pos_sim.unsqueeze(0), neg_sims], dim=0)  # (1 + num_neg, 1)
    labels = torch.zeros(m_norm.size(0), dtype=torch.long, device=m_new.device)  # 正样本 index=0 (1, )
    contrastive_loss = F.cross_entropy(logits.T, labels) # () scalar
    # 混合 loss
    return alpha * mse_loss + (1 - alpha) * contrastive_loss


def compute_target_loss(m_new, target_embed, negatives=None, alpha=1.0):
    # 直接 MSE，不 normalize
    mse_loss = F.mse_loss(m_new, target_embed)

    if negatives is not None:
        neg_loss = F.mse_loss(m_new.expand_as(negatives), negatives, reduction='mean')
        return mse_loss + 0.1 * neg_loss  # 小权重推开负样本
    return mse_loss


class InjectionCache(Cache):
    def __init__(self, memory_kv):
        self.key_cache = [layer[0] for layer in memory_kv]
        self.value_cache = [layer[1] for layer in memory_kv]
        
        # 你的逻辑：初始逻辑长度为 0
        self._seen_tokens = 0 
        self.memory_len = self.key_cache[0].shape[2] 
        
        self._is_compiled = False
        self._is_quantized = False
        self.has_layers = True
        self.layers = [None] * len(self.key_cache)

    @property
    def is_compileable(self):
        return False

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._seen_tokens

    # 【新增方法】修复 IndexError: index -1
    def get_max_cache_shape(self):
        # 返回格式为 (batch_size, num_heads, max_seq_len, head_dim)
        # 只要 max_seq_len 够大就行，这里我们根据当前物理长度返回
        sample_k = self.key_cache[0]
        return (sample_k.shape[0], sample_k.shape[1], 8192, sample_k.shape[3])

    def get_mask_sizes(self, cache_position, layer_idx=0):
        # 物理偏移 = memory 长度 + 已生成的 query 长度
        kv_offset = self.memory_len + self._seen_tokens
        kv_length = kv_offset + cache_position.shape[0]
        return kv_length, kv_offset

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
        self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=2)
        if layer_idx == len(self.key_cache) - 1:
            self._seen_tokens += key_states.shape[2]
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_usable_length(self, seq_len, layer_idx=0):
        return self.memory_len + self._seen_tokens

    def __getitem__(self, idx): return (self.key_cache[idx], self.value_cache[idx])
    def __len__(self): return len(self.key_cache)

class KVProjector(nn.Module):
    def __init__(self, config, use_layernorm=True):
        super().__init__()
        self.num_layers = config.num_hidden_layers
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.use_layernorm = use_layernorm

        self.projections = nn.ModuleList()
        for _ in range(self.num_layers):
            mlp = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size * 2),
                nn.GELU(),
                nn.Linear(self.hidden_size * 2, 2 * self.num_kv_heads * self.head_dim)
            )
            self.projections.append(mlp)

        if use_layernorm:
            self.ln = nn.LayerNorm(self.hidden_size)

    def forward(self, m_t: torch.Tensor):
        """
        m_t: (B, 1, hidden_size)  只接受单个状态向量
        返回: tuple of (k, v) per layer, each (B, num_kv_heads, 1, head_dim)
        """
        B = m_t.size(0)
        if self.use_layernorm:
            m_t = self.ln(m_t)

        past_key_values = []
        for mlp in self.projections:
            kv_flat = mlp(m_t)  # (B, 1, 2 * num_kv_heads * head_dim)
            k, v = kv_flat.chunk(2, dim=-1)
            k = k.view(B, self.num_kv_heads, 1, self.head_dim)
            v = v.view(B, self.num_kv_heads, 1, self.head_dim)
            past_key_values.append((k, v))
        return tuple(past_key_values)


class SlotSelector(nn.Module):
    def __init__(self, hidden_size, history_len, dropout=0.1, ordinal_scale=2.0):
        super().__init__()
        self.ordinal_scale = ordinal_scale
        self.history_len = history_len
        self.hidden_size = hidden_size

        # 融合当前 turn 的 r_agg 和历史每个槽位 m_read
        # 先做 linear 投影，再相加
        self.r_proj = nn.Linear(hidden_size, hidden_size)
        self.m_proj = nn.Linear(hidden_size, hidden_size)

        # 输出每个 slot 的 logits
        self.output_content = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

        self.output_ordinal = nn.Linear(hidden_size, 1)

    def forward(
        self, 
        r_agg: torch.Tensor, 
        m_read: torch.Tensor,
        time_emb: torch.Tensor,
    ):
        """
        r_agg: (B, 1, H)
        m_read: (B, history_len, H)
        time_emb: (B, history_len, H)
        返回:
            slot_logits: (B, history_len)
        """
        r = self.r_proj(r_agg)          # (B,1,H)
        m = self.m_proj(m_read)         # (B,L,H)
        content_h = r + m               # (B,L,H)
        content_logits = self.output_content(content_h).squeeze(-1)   # (B,L)
        ordinal_logits = self.output_ordinal(time_emb).squeeze(-1)    # (B,L)
        # ===== 强制加权 =====
        slot_logits = content_logits + self.ordinal_scale * ordinal_logits
        return slot_logits


def _init_transformer_style(module, std=0.02):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def init_kv_projector(module, std=0.01):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class CASSKVInjectionModel(nn.Module):
    def __init__(self, model_id):
        super().__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        self.config = self.base_model.config
        self.hidden_size = self.config.hidden_size
        self.history_len = 8

        dtype = self.base_model.dtype
        self.m_0 = nn.Parameter(torch.randn(1, self.history_len, self.hidden_size, dtype=dtype))
        nn.init.normal_(self.m_0, std=0.02)

        # 两个独立的 projector：一个给“最新状态”（强权威），一个给“检索到的历史状态”
        self.latest_projector = KVProjector(self.config).to(dtype=dtype)   # 专用于最新状态
        self.history_projector = KVProjector(self.config).to(dtype=dtype)  # 专用于检索状态

        self.probe = nn.Linear(2 * self.hidden_size, 1, dtype=dtype)

        self.synthesis = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        ).to(dtype=dtype)

        # 历史位置选择器（关键！用于历史查询）
        self.slot_selector = SlotSelector(
            hidden_size=self.hidden_size, history_len=self.history_len
        ).to(dtype)

        self.time_emb = nn.Embedding(self.history_len, self.hidden_size, dtype=dtype)
        nn.init.normal_(self.time_emb.weight, std=0.02)

        self.ordinal_emb = nn.Embedding(self.history_len, self.hidden_size, dtype=dtype)
        nn.init.normal_(self.ordinal_emb.weight, std=0.02)
        
        self.read_content_proj = nn.Linear(self.hidden_size, self.hidden_size, dtype=dtype)
        
        init_kv_projector(self.latest_projector, 0.01)
        init_kv_projector(self.history_projector, 0.01)
        _init_transformer_style(self.probe, 0.01)
        _init_transformer_style(self.synthesis)
        _init_transformer_style(self.slot_selector)
        _init_transformer_style(self.read_content_proj)

    
    @property
    def device(self):
        return self.m_0.device
    
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        m_prev: torch.Tensor,           # (1, history_len, hidden)
        current_len: torch.LongTensor,  # (1,) 当前有效长度
        update_flag: torch.FloatTensor, # 1.0 表示需要更新最新状态
        target_embed: torch.Tensor = None,     # update turn 用的权威嵌入
        negative_embeds: List[torch.Tensor] = None,
        slot_label: torch.LongTensor = None,   # query turn 用的历史位置标签
    ):
        """
        input_ids: 1,L
        labels: 1,L
        m_prev: 1,8,H
        current_len: 1,
        update_flag: (), scalar
        target_embed: 1,H
        negative_embeds: num_neg, H
        slot_label: (), scalar
        
        返回 loss_dict 和新的 m_new, new_len
        """
        B = 1  # 固定 batch=1
        device = input_ids.device
        m_prev = m_prev.to(torch.bfloat16)
        current_len = current_len.clamp(max=self.history_len)

        # ================= 1. 当前 turn 表示 =================
        token_embeds = self.base_model.get_input_embeddings()(input_ids)
        outputs = self.base_model.model(inputs_embeds=token_embeds, use_cache=False)
        r_agg = outputs.last_hidden_state[:, -1, :]  # (1, H)

        loss_dict = {
            "llm_loss": torch.tensor(0.0, device=device),
            "target_loss": torch.tensor(0.0, device=device),
            "probe_loss": torch.tensor(0.0, device=device),
            "select_loss": torch.tensor(0.0, device=device),
            "latest_attn_loss": torch.tensor(0.0, device=device),
        }

        # ================= 2. 更新最新状态 =================
        latest_m = m_prev[:, -1, :]  # 当前最新状态 (1, H)
        g_logits = self.probe(torch.cat([r_agg, latest_m], dim=-1)) # (1,1)
        probe_loss = F.binary_cross_entropy_with_logits(g_logits.squeeze(-1), update_flag.unsqueeze(-1)) # (1, ), (1, )
        loss_dict["probe_loss"] = probe_loss # () scalar

        update_index = torch.clamp(current_len - 1, min=0, max=self.history_len - 1)
        ordinal = self.ordinal_emb(update_index)

        # Teacher forcing：使用真实 update_flag 更新记忆（训练稳定）
        g_signal = update_flag.unsqueeze(-1)  # (1,)
        m_candidate = self.synthesis(torch.cat([r_agg, latest_m], dim=-1)) # (1, H)
        m_new_latest = g_signal * (m_candidate + ordinal) + (1 - g_signal) * latest_m # (1, H)

        # 更新队列：向右 roll，最新位置写入新状态
        m_new = torch.roll(m_prev, shifts=-1, dims=1) # (1, history_len, H)
        m_new[:, -1, :] = m_new_latest

        # 更新有效长度
        new_len = current_len.clone() # (1, )
        if update_flag.item() > 0.5:
            new_len = torch.min(current_len + 1, torch.tensor(self.history_len, device=device))

        # target 对齐损失（仅 update turn）
        if update_flag.item() > 0.5 and target_embed is not None:
            target_loss = compute_target_loss(
                m_new_latest,   # (1, H)
                target_embed,   # (1, H)
                negatives=negative_embeds, # (num_neg, H)
                alpha=0.5
            ) # scalar
            loss_dict["target_loss"] = target_loss

        # ================= 3. Query turn：slot 选择 + KV 注入 =================
        if update_flag.item() < 0.5:
            positions = torch.arange(self.history_len, device=device)  # (history_len,)
            time_bias = self.time_emb(positions).unsqueeze(0)                        # (history_len, H)
            m_read = (
                m_new.detach() 
                # + 0.5 * self.read_content_proj(m_new.detach())
            )
            # (1, history_len, H)

            slot_logits = self.slot_selector(r_agg.unsqueeze(1), m_read, time_bias)  # (1, history_len)
            # 只在前 current_len 个位置上计算 softmax（防止无效位置干扰）
            mask = torch.arange(self.history_len, device=device).unsqueeze(0) < current_len
            slot_logits = slot_logits.masked_fill(~mask, -1e9)
            # slot_logits = slot_logits / 8.0
            slot_probs = F.softmax(slot_logits, dim=-1)
            print(f"slot_prob: {[f'{i:.4f}' for i in slot_probs[0]]}, index: {slot_probs.argmax().item()}, slot_label: {slot_label}")

            # 训练 loss
            if slot_label is not None:
                if slot_label.dim() == 0:
                    slot_label = slot_label.unsqueeze(0)
                assert slot_label.max() < self.history_len and slot_label.min() >= 0
                select_loss = F.cross_entropy(slot_logits, slot_label)
                latest_attn_loss = -slot_probs[:, -1].log().mean() * 0.1
                loss_dict["select_loss"] = select_loss
                loss_dict["latest_attn_loss"] = latest_attn_loss

            # ============== 生成阶段（注入两个虚拟 KV）==============
            # a. 始终注入最新状态（强权威）
            latest_kv = self.latest_projector(m_new[:, -1:, :])  # (1,1,D) → per-layer KV ((k,v), ...)
            # b. 注入检索到的历史状态（软选择）
            selected_m = torch.sum(m_new * slot_probs.unsqueeze(-1), dim=1, keepdim=True)  # (1,1,history_len) -> 1,8,H
            history_kv = self.history_projector(selected_m) #  1,8,H -> per-layer KV ((k,v), ...)

            # 合并 KV
            memory_kv = []
            for layer_idx in range(self.base_model.config.num_hidden_layers):
                k_latest, v_latest = latest_kv[layer_idx]
                k_hist, v_hist = history_kv[layer_idx]
                k_comb = torch.cat([k_latest, k_hist], dim=2)
                v_comb = torch.cat([v_latest, v_hist], dim=2)
                memory_kv.append((k_comb, v_comb))
            memory_cache = InjectionCache(memory_kv)
            L = input_ids.size(1)
            position_ids = torch.arange(0, L, device=device).unsqueeze(0)
            outputs_gen = self.base_model.model(
                input_ids=input_ids,
                position_ids=position_ids,
                past_key_values=memory_cache,
                use_cache=False,
            )

            # past_key_values = DynamicCache()
            # for layer_idx in range(self.config.num_hidden_layers):
            #     # 最新状态 KV + 历史状态 KV → 长度为2的虚拟 prefix
            #     k_latest, v_latest = latest_kv[layer_idx]
            #     k_hist, v_hist = history_kv[layer_idx]
            #     k_combined = torch.cat([k_latest, k_hist], dim=2)  # (1, heads, 2, head_dim) (1,4,2,128)
            #     v_combined = torch.cat([v_latest, v_hist], dim=2)
            #     past_key_values.update(k_combined, v_combined, layer_idx)

            # past_len = past_key_values.get_seq_length()  # = 2
            # position_ids = torch.arange(
            #     past_len,
            #     past_len + input_ids.size(1),
            #     device=input_ids.device
            # ).unsqueeze(0)
            # # 再次 forward 生成 logits
            # outputs_gen = self.base_model.model(
            #     # inputs_embeds=token_embeds,
            #     input_ids=input_ids,
            #     position_ids=position_ids,
            #     past_key_values=past_key_values,
            #     use_cache=False,
            #     output_attentions=True,
            # )

            logits = self.base_model.lm_head(outputs_gen.last_hidden_state)
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = logits[..., :-1, :].contiguous()
            llm_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
            loss_dict["llm_loss"] = llm_loss

        return loss_dict, m_new.detach(), new_len, g_logits