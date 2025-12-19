import random
import numpy as np

# 示例：每个领域 slot 的值扩充到 6 个
SINGLE_SLOT_DOMAINS_EXTENDED = {
    "跨国差旅": {"slot": "签证进度", "values": ["材料已提交", "补件中", "已出签", "延期审核", "加急处理中", "签证过期"]},
    "智慧健康": {"slot": "心率预警值", "values": ["100bpm", "110bpm", "120bpm", "130bpm", "140bpm", "150bpm"]},
    "游戏数值": {"slot": "角色等级", "values": ["35", "36", "37", "38", "39", "40"]},
    "法律咨询": {"slot": "证据状态", "values": ["齐全", "部分缺失", "待补充", "已补充", "证据作废", "证据待鉴定"]},
    "智能厨房": {"slot": "当前温度", "values": ["180°C", "190°C", "200°C", "210°C", "220°C", "烹饪结束"]},
    "投资理财": {"slot": "持仓比例", "values": ["10%", "20%", "30%", "50%", "70%", "80%"]},
    "软件开发": {"slot": "代码状态", "values": ["未提交", "已自测", "待评审", "已合入", "回滚", "冲突待处理"]},
    "教育辅导": {"slot": "错题数量", "values": ["3道", "5道", "8道", "10道", "12道", "15道"]},
    "心理咨询": {"slot": "当前情绪", "values": ["焦虑", "低落", "愤怒", "平静", "轻松", "紧张"]},
    "物流调度": {"slot": "当前载重", "values": ["5吨", "8吨", "12吨", "15吨", "18吨", "已卸货"]},
    "社交约会": {"slot": "惊喜程度", "values": ["无", "小惊喜", "精心准备", "完成", "意外惊喜", "延期"]},
}

def generate_single_slot_dialogue_extended(dialog_id):
    domain_key = random.choice(list(SINGLE_SLOT_DOMAINS_EXTENDED.keys()))
    config = SINGLE_SLOT_DOMAINS_EXTENDED[domain_key]
    
    slot = config["slot"]
    current_val = "None"
    turns = []

    # 6-8轮对话
    for i in range(random.randint(6, 8)):
        rand_val = random.random()
        update_flag = 0

        if rand_val < 0.35:  # 新增状态
            new_val = random.choice(config["values"])
            user_input = f"现在把{slot}更新为{new_val}。"
            current_val = new_val
            llm_output = f"好的，{domain_key}的{slot}现在已记录为{new_val}。"
            update_flag = 1

        elif rand_val < 0.6:  # 否定并重设
            old_val = current_val
            new_val = random.choice(config["values"])
            user_input = f"之前说的{slot}是{old_val}，那个不对，改成{new_val}。"
            current_val = new_val
            llm_output = f"已修正，废弃{old_val}，目前{slot}设定为{new_val}。"
            update_flag = 1

        else:  # 回溯查询
            user_input = f"在最开始的时候，我的{slot}设置的是什么？"
            llm_output = f"通过回顾，您最开始对{slot}的操作记录在对话历史中，目前的设定是{current_val}。"
            update_flag = 0

        state_desc = f"{slot}:{current_val}"
        
        in_turn_negatives = [v for v in config["values"] if v != current_val]
        all_out_turn_negatives = []
        for key, val in SINGLE_SLOT_DOMAINS_EXTENDED.items():
            if key == domain_key:
                continue
            val_list = val["values"]
            all_out_turn_negatives.extend(val_list)
        out_turn_negatives = np.random.choice(all_out_turn_negatives, 8 - len(in_turn_negatives))

        negatives = in_turn_negatives + out_turn_negatives.tolist()
        negatives = [f"{slot}:{v}" for v in negatives]

        turns.append({
            "turn_id": i + 1,
            "user_input": user_input,
            "update_flag": update_flag,
            "state_desc": state_desc,
            "llm_output": llm_output,
            "negative_state_desc": negatives,
        })

    return {
        "dialogue_id": f"SINGLE_{dialog_id}",
        "turns": turns,
        "topic": domain_key
    }

# 示例调用
# dialogue = generate_single_slot_dialogue_extended(1)
# print(dialogue)


import pnlp
res = []
for i in range(2000):
    dialogue = generate_single_slot_dialogue_extended(i)
    res.append(dialogue)


pnlp.write_list_dict_to_file("kv_mock_singleslot_2000.jsonl", res)
