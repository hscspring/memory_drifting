import json
import random
from typing import List

# ====== 多主题配置（不变）======
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
    1: "第一个",
    2: "第二个",
    3: "第三个",
    4: "第四个",
    5: "第五个",
    6: "第六个",
}

NUM_DIALOGUES = 1000
MIN_WRITE_TURNS = 4
MAX_WRITE_TURNS = 8        # 稍微加长一点，支持更深的相对查询
MAX_QUERY_OFFSET = 5       # 最多问“上上上上上一次”（-5），避免太难

# ====== 工具函数（不变）======
def sample_negatives(values: List[str], pos: str, k=3):
    pos_idx = values.index(pos)
    neighbors = []
    if pos_idx - 1 >= 0:
        neighbors.append(values[pos_idx - 1])
    if pos_idx + 1 < len(values):
        neighbors.append(values[pos_idx + 1])
    far = [v for i, v in enumerate(values) if abs(i - pos_idx) >= 2]
    negatives = []
    if neighbors:
        negatives.extend(random.sample(neighbors, k=min(1, len(neighbors))))
    if far and len(negatives) < k:
        negatives.extend(random.sample(far, k=min(k - len(negatives), len(far))))
    # 如果还不够，随机补
    if len(negatives) < k:
        pool = [v for v in values if v != pos]
        negatives.extend(random.sample(pool, k=k - len(negatives)))
    return negatives

def gen_write_turn(turn_id, domain, slot, value, values):
    return {
        "turn_id": turn_id,
        "user_input": f"{domain}场景下，当前{slot}是{value}。",
        "llm_output": f"已记录，{slot}更新为{value}。",
        "update_flag": 1,
        "state_desc": value,
        "negative_state_desc": sample_negatives(values, value),
        # 新字段：用于监督 offset 预测（这里 None，因为是更新轮）
        "relative_offset": None,
    }

# ====== 新增：序数查询（第X次）→ 转换为相对偏移 ======
def gen_ordinal_query(turn_id, domain, slot, history):
    if len(history) <= 1:
        return None
    # 限制最多问到“第六次”，避免太远的历史
    max_n = min(6, len(history) - 1)
    if max_n < 1:
        return None
    n = random.randint(1, max_n)       # 第1次 到 第max_n次
    idx = n - 1                        # 对应 history 中的索引
    target_value = history[idx]

    # 动态生成序数词
    if n == 1:
        ordinal = "第一次"
    elif n == 2:
        ordinal = "第二次"
    elif n == 3:
        ordinal = "第三次"
    else:
        ordinal = f"第{n}次"

    relative_offset = -(len(history) - idx - 1)  # 正确计算倒数偏移

    return {
        "turn_id": turn_id,
        "user_input": f"在{domain}中，你还记得我{ordinal}提到的{slot}是多少吗？",
        "llm_output": f"你{ordinal}提到的{slot}是{target_value}。",
        "update_flag": 0,
        "state_desc": "",
        "negative_state_desc": [],
        "relative_offset": relative_offset,
    }

# ====== 相对查询（在我说XX之前，上一次是什么）======
def gen_relative_query(turn_id, domain, slot, history):
    # 必须至少有2次更新
    if len(history) < 2:
        return None

    # 随机选一个参考点（不能选最早的）
    ref_idx = random.randint(1, len(history) - 1)
    ref_value = history[ref_idx]
    target_idx = ref_idx - 1
    target_value = history[target_idx]

    # 相对偏移永远是 -1（因为是“上一次”）
    relative_offset = -1

    return {
        "turn_id": turn_id,
        "user_input": f"在我说{slot}是{ref_value}之前，上一次的{slot}是什么？",
        "llm_output": f"在{ref_value}之前，{slot}是{target_value}。",
        "update_flag": 0,
        "state_desc": "",
        "negative_state_desc": [],
        "relative_offset": relative_offset,
    }

# ====== 额外新增：更灵活的“上X次”查询（增加多样性）======
def gen_upper_query(turn_id, domain, slot, history):
    if len(history) < 3:
        return None
    # 随机选偏移 2~5（上上一次 到 上上上上上一次）
    offset = random.randint(2, min(MAX_QUERY_OFFSET, len(history) - 1))
    target_idx = len(history) - offset  # 正确：最新是 -1，减 offset
    target_value = history[target_idx]

    offset_words = ["上上", "上上上", "上上上上", "上上上上上"][offset-2]

    return {
        "turn_id": turn_id,
        "user_input": f"{domain}场景下，{offset_words}一次的{slot}是什么？",
        "llm_output": f"{offset_words}一次的{slot}是{target_value}。",
        "update_flag": 0,
        "state_desc": "",
        "negative_state_desc": [],
        "relative_offset": -offset,
    }

# ====== 主生成逻辑（关键修改）======
def generate_dialogue(dialogue_id: int):
    domain = random.choice(list(SINGLE_SLOT_DOMAINS_EXTENDED.keys()))
    cfg = SINGLE_SLOT_DOMAINS_EXTENDED[domain]
    slot = cfg["slot"]
    values = cfg["values"]

    num_writes = random.randint(MIN_WRITE_TURNS, MAX_WRITE_TURNS)
    # 有放回抽样，允许重复
    chosen_values = [random.choice(values) for _ in range(num_writes)]

    turns = []
    history = []
    turn_id = 1

    # 1. 先全部生成写入轮
    for v in chosen_values:
        turns.append(gen_write_turn(turn_id, domain, slot, v, values))
        history.append(v)
        turn_id += 1

    # 2. 现在基于完整的 history 生成查询轮（turn_id 继续递增，不打乱）
    queries = []

    # 序数查询（第X次）
    if len(history) > 1 and random.random() < 0.6:
        q = gen_ordinal_query(turn_id, domain, slot, history)
        if q:
            queries.append(q)
            turn_id += 1

    # 经典“在XX之前，上一次”
    if len(history) >= 2 and random.random() < 0.8:
        q = gen_relative_query(turn_id, domain, slot, history)
        if q:
            queries.append(q)
            turn_id += 1

    # “上X次”查询
    if len(history) >= 3 and random.random() < 0.7:
        q = gen_upper_query(turn_id, domain, slot, history)
        if q:
            queries.append(q)
            turn_id += 1

    # 可选：随机打乱查询顺序，但不影响 turn_id（turn_id 已分配好）
    random.shuffle(queries)

    # 3. 添加查询到 turns
    turns.extend(queries)

    return {
        "dialogue_id": f"dlg_{dialogue_id:06d}",
        "domain": domain,
        "slot": slot,
        "turns": turns,
        "history_values": history,
    }

def main():
    data = [generate_dialogue(i) for i in range(NUM_DIALOGUES)]
    with open("mock_dialogues_multi_domain_v2.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Generated {len(data)} dialogues -> mock_dialogues_multi_domain_v2.json")
    print("新特性：")
    print("  - 查询标签改为 relative_offset（例如 -1, -2, -3...）")
    print("  - 支持序数、第X次、上X次、在XX之前等多种查询形式")
    print("  - 适合训练轻量 offset 预测头")

if __name__ == "__main__":
    main()