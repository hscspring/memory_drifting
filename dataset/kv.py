import torch
import copy
from dataclasses import dataclass
from typing import Dict, List, Any


@dataclass
class CASSKVSequentialCollator:
    tokenizer: Any
    max_length: int = 1024

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        features: 这里的 features 是一个 Batch 的对话列表。
        因为 CASS 需要跨轮次传递 M_t，建议训练时 batch_size=1，
        即一次处理一个 dialogue_id 下的所有 turns。
        """
        # 我们假设 batch_size = 1
        dialogue = features[0]
        turns_data = []

        for turn in dialogue["turns"]:
            # 1. 构造标准对话格式
            # 我们需要两个版本：一个是到 User 为止（用于计算 Mask），一个是完整的（用于训练）
            user_msg = [{"role": "user", "content": turn["user_input"]}]
            assistant_msg = {"role": "assistant", "content": turn["llm_output"]}
            
            # 使用模板生成文本
            # prompt_text: <|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n
            prompt_text = self.tokenizer.apply_chat_template(
                user_msg, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # full_text: <|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>\n
            full_messages = user_msg + [assistant_msg]
            full_text = self.tokenizer.apply_chat_template(
                full_messages, 
                tokenize=False
            )

            # 2. Tokenize
            prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
            full_ids = self.tokenizer.encode(full_text, add_special_tokens=False)
            
            # 3. 构造 Labels 并进行 Masking
            # 我们不希望模型学习 User 的输入，所以把 User 部分的 Label 设为 -100
            labels = copy.deepcopy(full_ids)
            prompt_len = len(prompt_ids)
            for i in range(prompt_len):
                labels[i] = -100 # PyTorch CrossEntropyLoss 会自动忽略 -100
                
            # 截断处理
            full_ids = full_ids[:self.max_length]
            labels = labels[:self.max_length]

            # 4. 预处理状态描述 (用于生成 E_target)
            # state_desc 不需要模板，直接编码即可
            target_ids = self.tokenizer.encode(
                turn["state_desc"], 
                add_special_tokens=False, 
                max_length=128, 
                truncation=True
            )
            negatives_ids: List[List[int]] = self.tokenizer(turn["negative_state_desc"], 
                add_special_tokens=False, 
                max_length=128, 
                truncation=True, 
                padding=True, 
                padding_side="left"
            )["input_ids"]
            turns_data.append({
                "input_ids": torch.tensor(full_ids),
                "labels": torch.tensor(labels),
                "update_flag": torch.tensor(turn["update_flag"]).float(),
                "target_ids": torch.tensor(target_ids),
                "negative_state_ids": torch.tensor(negatives_ids),
            })

        return {"dialogue_turns": turns_data}