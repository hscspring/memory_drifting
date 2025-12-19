import os

import torch


def save_checkpoint(model, optimizer, path, step):
    """
    只保存可训练的参数 (KVProjector + Probe + Synthesis + m_0)
    """
    checkpoint_dir = os.path.join(path, f"step_{step}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 提取可训练参数的状态字典
    trainable_state_dict = {
        n: p for n, p in model.named_parameters() if p.requires_grad
    }
    
    torch.save(trainable_state_dict, os.path.join(checkpoint_dir, "cass_adapter.pt"))
    print(f"\nCheckpoint saved at step {step}")