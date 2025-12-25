import torch
from dataclasses import dataclass
from typing import Dict, List, Any


"""
Data Protocol:
- update turn:
    - update_flag = 1
    - target_ids != None
    - negative_state_ids != None
    - slot_label = None
- query / passive turn:
    - update_flag = 0
    - target_ids = None
    - negative_state_ids = None
    - slot_label != None
"""


@dataclass
class CASSKVSequentialCollator:
    tokenizer: Any
    max_length: int = 2048
    target_max_length: int = 128  # state_desc 的最大长度

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        features: List of dialogues, 但我们只取第一个（batch_size=1）
        返回一个包含所有 turns 的列表，便于 train_one_dialogue 顺序处理
        """
        assert len(features) == 1, "CASS collator 只支持 batch_size=1"
        dialogue = features[0]
        turns_data = []

        for turn in dialogue["dialogue_turns"]:
            update_flag = int(turn["update_flag"])
            user_msg = [{"role": "user", "content": turn["user_input"]}]
            assistant_msg = {"role": "assistant", "content": turn["llm_output"]}

            if update_flag == 1:
                # === update turn: 只给 user 输入，assistant 不参与标签 ===
                prompt_text = self.tokenizer.apply_chat_template(
                    user_msg, tokenize=False, add_generation_prompt=False
                )
                full_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
                labels = [-100] * len(full_ids)  # 全掩码

                # 权威状态描述（正样本）
                target_ids = self.tokenizer.encode(
                    turn["state_desc"],
                    add_special_tokens=False,
                    max_length=self.target_max_length,
                    truncation=True,
                )
                target_ids = torch.tensor(target_ids, dtype=torch.long)  # (seq_len,)

                # 负样本状态描述
                negative_state_descs: List[str] = turn.get("negative_state_desc", [])
                if negative_state_descs:
                    neg_tokenized = self.tokenizer(
                        negative_state_descs,
                        add_special_tokens=False,
                        max_length=self.target_max_length,
                        truncation=True,
                        padding="max_length",          # 统一长度，便于 batch
                        padding_side="left",
                        return_tensors="pt",
                    )
                    negative_state_ids = neg_tokenized["input_ids"]  # (num_neg, target_max_len)
                else:
                    negative_state_ids = None

                slot_label = None

            else:
                # === query turn: user + assistant 全序列 ===
                full_messages = user_msg + [assistant_msg]
                full_text = self.tokenizer.apply_chat_template(full_messages, tokenize=False)
                full_ids = self.tokenizer.encode(full_text, add_special_tokens=False)

                # prompt 部分（仅 user）用于计算 labels 掩码
                prompt_text = self.tokenizer.apply_chat_template(
                    user_msg, tokenize=False, add_generation_prompt=True
                )
                prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
                prompt_len = len(prompt_ids)

                labels = [-100] * prompt_len + full_ids[prompt_len:]

                # 截断
                full_ids = full_ids[:self.max_length]
                labels = labels[:self.max_length]

                target_ids = None
                negative_state_ids = None
                slot_label = (
                    torch.tensor(turn["slot_label"], dtype=torch.long)
                    if turn["slot_label"] is not None else None
                )

            turns_data.append({
                "input_ids": torch.tensor(full_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "update_flag": torch.tensor(update_flag, dtype=torch.float),
                "target_ids": target_ids,              # (seq_len,) 或 None
                "negative_state_ids": negative_state_ids,  # (num_neg, seq_len) 或 None
                "slot_label": slot_label,              # tensor scalar 或 None
            })

        return {"dialogue_turns": turns_data}