import json
import random

EXPANDED_DOMAINS = {

    # 1. 跨国差旅（专有名词 + 数值限制）
    "跨国差旅": {
        "slots": ["签证进度", "接送机地点", "结算币种", "报销上限"],
        "values": [
            "进度:材料已提交/补件中/已出签",
            "地点:成田机场T1/浦东机场T2/希思罗T5",
            "币种:CNY/USD/JPY/EUR",
            "上限:3000元/500美元/8万日元"
        ]
    },

    # 2. 智慧健康（低容错精度）
    "智慧健康": {
        "slots": ["药物剂量", "过敏史", "心率预警值", "医嘱状态"],
        "values": [
            "剂量:5mg/10mg/20mg",
            "过敏:青霉素/花生/无",
            "心率:120bpm/140bpm",
            "医嘱:按时服用/需复诊/暂停用药"
        ]
    },

    # 3. 游戏数值（频繁小幅变化）
    "游戏数值": {
        "slots": ["角色等级", "属性分配", "装备耐久", "当前副本"],
        "values": [
            "等级:35/36/37",
            "加点:力量20/敏捷15/智力10",
            "耐久:92%/75%/40%",
            "副本:熔岩遗迹/暗影地宫/天空之塔"
        ]
    },

    # 4. 法律咨询（严肃语义 + 逻辑链）
    "法律咨询": {
        "slots": ["案件类型", "涉案金额", "起诉期限", "证据状态"],
        "values": [
            "类型:劳动仲裁/合同纠纷/侵权责任",
            "金额:5万/20万/100万",
            "期限:30天内/6个月内/已过期",
            "证据:齐全/部分缺失/待补充"
        ]
    },

    # 5. 智能厨房（物理参数 + 偏好）
    "智能厨房": {
        "slots": ["模式", "当前温度", "剩余时长", "食材"],
        "values": [
            "模式:烘焙/空炸/解冻",
            "温度:180°C/200°C/220°C",
            "时长:10min/25min/40min",
            "食材:牛肉/土豆/鸡翅"
        ]
    },

    # 6. 投资理财（比例与目标动态）
    "投资理财": {
        "slots": ["标的名称", "持仓比例", "止损位", "风险偏好"],
        "values": [
            "标的:黄金/白银/纳指",
            "比例:20%/50%/80%",
            "止损:10%/15%/20%",
            "偏好:激进/稳健/保守"
        ]
    },

    # 7. 软件开发（流程化状态切换）
    "软件开发": {
        "slots": ["当前分支", "代码状态", "优先级", "负责人"],
        "values": [
            "分支:main/dev/feature-1",
            "状态:未提交/已自测/待评审",
            "优先级:P0/P1/P2",
            "负责人:张三/李四/王五"
        ]
    },

    # 8. 教育辅导（抽象程度）
    "教育辅导": {
        "slots": ["学习进度", "错题数量", "下次重点", "掌握程度"],
        "values": [
            "进度:第三章/期中复习/函数单元",
            "错题:3道/8道/15道",
            "重点:应用题/概念理解/计算能力",
            "掌握:熟练/一般/薄弱"
        ]
    },

    # 9. 心理咨询（非结构化情绪压缩）
    "心理咨询": {
        "slots": ["当前情绪", "压力来源", "干预策略", "谈话重点"],
        "values": [
            "情绪:焦虑/低落/愤怒",
            "压力:工作/家庭/人际关系",
            "策略:情绪觉察/放松训练/认知重构",
            "重点:自我价值/边界感/情绪表达"
        ]
    },

    # 10. 物流调度（强时序 + 空间）
    "物流调度": {
        "slots": ["货品类型", "当前载重", "运输节点", "预计到达"],
        "values": [
            "货品:冷链食品/电子产品/建材",
            "载重:8吨/12吨/18吨",
            "轨迹:上海-南京-合肥/广州-长沙",
            "到达:2小时/6小时/明日12点"
        ]
    },

    # 11. 社交约会（软性偏好管理）
    "社交约会": {
        "slots": ["约会地点", "穿搭风格", "禁忌话题", "惊喜程度"],
        "values": [
            "地点:咖啡馆/日料店/公园",
            "穿搭:休闲/正式/运动风",
            "话题:前任/收入/家庭矛盾",
            "惊喜:无/小惊喜/精心准备"
        ]
    }
}


def generate_ultra_diverse_dialogue(dialog_id):
    domain_key = random.choice(list(EXPANDED_DOMAINS.keys()))
    config = EXPANDED_DOMAINS[domain_key]
    
    current_slots = {slot: "None" for slot in config["slots"]}
    turns = []
    
    # 模拟更长、更复杂的对话逻辑 (8-15轮)
    for i in range(random.randint(8, 15)):
        update_flag = 0
        slot = random.choice(config["slots"])
        
        # 随机三种行为：新设、否定重设、历史追溯
        rand_val = random.random()
        if rand_val < 0.4: # 新增状态
            new_val = random.choice(config["values"])
            user_input = f"现在把{slot}更新为{new_val}。"
            current_slots[slot] = new_val
            llm_output = f"好的，{domain_key}的{slot}现在已记录为{new_val}。"
            update_flag = 1
        elif rand_val < 0.7: # 否定并重设（高干扰）
            old_val = current_slots[slot]
            new_val = random.choice(config["values"])
            user_input = f"之前说的{slot}是{old_val}，那个不对，改成{new_val}。"
            current_slots[slot] = new_val
            llm_output = f"已修正，废弃{old_val}，目前{slot}设定为{new_val}。"
            update_flag = 1
        else: # 纯回溯（不更新状态）
            user_input = f"在最开始的时候，我的{slot}设置的是什么？"
            llm_output = f"通过回顾，您最开始对{slot}的操作记录在对话历史中，目前的设定是{current_slots[slot]}。"
            update_flag = 0

        state_desc = f"[{domain_key}] " + " | ".join([f"{k}:{v}" for k, v in current_slots.items()])
        
        turns.append({
            "turn_id": i + 1,
            "user_input": user_input,
            "update_flag": update_flag,
            "state_desc": state_desc,
            "llm_output": llm_output
        })
    return {
        "dialogue_id": f"ULTRA_{dialog_id}", 
        "turns": turns,
        "topic": domain_key,
    }



import pnlp
res = []
for i in range(2000):
    dialogue = generate_ultra_diverse_dialogue(i)
    res.append(dialogue)


pnlp.write_list_dict_to_file("kv_mock_2000.jsonl", res)