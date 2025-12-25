import json
import sys
import random
from typing import List


# ====== 多主题配置 ======
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

ORDINAL_MAP = {
    1: "第一",
    2: "第二",
    3: "第三",
    4: "第四",
    5: "第五",
    6: "第六",
}

MIN_WRITE_TURNS = 4
MAX_WRITE_TURNS = 6


# ====== 工具函数 ======
def sample_negatives_simple(values: List[str], pos: str, k=3):
    pool = [v for v in values if v != pos]
    return random.sample(pool, k=min(k, len(pool)))


def sample_negatives(values, pos, k=3):
    pos_idx = values.index(pos)

    # 邻近值（hard）
    neighbors = []
    if pos_idx - 1 >= 0:
        neighbors.append(values[pos_idx - 1])
    if pos_idx + 1 < len(values):
        neighbors.append(values[pos_idx + 1])

    # 远值（easy）
    far = [v for i, v in enumerate(values) if abs(i - pos_idx) >= 2]

    negatives = []

    if neighbors:
        negatives.extend(random.sample(
            neighbors, k=min(1, len(neighbors))
        ))

    if far and len(negatives) < k:
        negatives.extend(random.sample(
            far, k=min(k - len(negatives), len(far))
        ))

    return negatives


def gen_write_turn(turn_id, domain, slot, value, values):
    return {
        "turn_id": turn_id,
        "user_input": f"{domain}场景下，当前{slot}是{value}。",
        "llm_output": f"已记录，{slot}更新为{value}。",
        "update_flag": 1,
        "state_desc": value,
        "negative_state_desc": sample_negatives(values, value),
        "slot_label": None,
    }


def gen_ordinal_query(turn_id, domain, slot, history, is_train):
    idx = random.randint(0, len(history) - 1)
    ordinal = ORDINAL_MAP[idx + 1]
    if is_train:
        llm_output =  f"你{ordinal}次提到的{slot}是{history[idx]}。"
    else:
        llm_output = history[idx]
    return {
        "turn_id": turn_id,
        "user_input": f"在{domain}中，你还记得我{ordinal}次提到的{slot}是多少吗？",
        "llm_output": llm_output,
        "update_flag": 0,
        "state_desc": "",
        "negative_state_desc": [],
        "slot_label": idx,
    }


def gen_relative_query(turn_id, domain, slot, history, is_train):
    ref_idx = random.randint(1, len(history) - 1)
    ref_value = history[ref_idx]
    target_idx = ref_idx - 1
    if is_train:
        llm_output = f"在{ref_value}之前，{slot}是{history[target_idx]}。"
    else:
        llm_output = history[target_idx]
    return {
        "turn_id": turn_id,
        "user_input": f"在我说{slot}是{ref_value}之前，上一次的{slot}是什么？",
        "llm_output": llm_output,
        "update_flag": 0,
        "state_desc": "",
        "negative_state_desc": [],
        "slot_label": target_idx,
    }


# ====== 主生成逻辑 ======
def generate_dialogue(dialogue_id: int, is_train: bool = True):
    domain = random.choice(list(SINGLE_SLOT_DOMAINS_EXTENDED.keys()))
    cfg = SINGLE_SLOT_DOMAINS_EXTENDED[domain]
    slot = cfg["slot"]
    values = cfg["values"]

    num_writes = random.randint(MIN_WRITE_TURNS, MAX_WRITE_TURNS)
    chosen_values = random.sample(values, num_writes)

    turns = []
    history = []
    turn_id = 1

    # 写入轮
    for v in chosen_values:
        turns.append(gen_write_turn(turn_id, domain, slot, v, values))
        history.append(v)
        turn_id += 1

    # 检索轮
    if random.random() < 0.5:
        turn = gen_ordinal_query(turn_id, domain, slot, history, is_train)
    else:
        turn = gen_relative_query(turn_id, domain, slot, history, is_train)
    if is_train:
        turns.append(turn)
        test_query = ""
        ground_truth = ""
        slot_label = None
    else:
        test_query = turn["user_input"]
        ground_truth = turn["llm_output"]
        slot_label = turn["slot_label"]

    return {
        "dialogue_id": f"dlg_{dialogue_id:06d}",
        "domain": domain,
        "slot": slot,
        "dialogue_turns": turns,
        "test_query": test_query,
        "ground_truth": ground_truth,
        "slot_label": slot_label
    }


def main():
    tag = int(sys.argv[1])
    num_dialogues = int(sys.argv[2])
    if tag == 1:
        is_train = True
    else:
        is_train = False
    
    data = [generate_dialogue(i, is_train) for i in range(num_dialogues)]
    out_file = f"mock_dialogues_multi_domain_istrain={is_train}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Generated {len(data)} dialogues -> {out_file}")


if __name__ == "__main__":
    main()
