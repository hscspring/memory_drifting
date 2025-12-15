from loguru import logger

from data_creator import do_llm_call


async def check_answer(response, ground_truth) -> bool:
    response_lower = response.strip().lower()
    ground_truth_lower = ground_truth.strip().lower()
    
    prompt = """You are a helpful assistant that judges whether the response is semantically consistent with the ground truth.
If the response contains all the information in the ground truth, answer "Yes", otherwise answer "No".
The output format is strictly one word: "Yes" or "No" without any additional text.
"""
    msgs = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Ground truth: {ground_truth_lower}\nResponse: {response_lower}"},
    ]
    output, reasoning = await do_llm_call("gpt-oss-120b", 0.2, msgs, "judger", 1024, False)
    logger.info(f"Judger output: {output}, reasoning: {reasoning}")
    return output.strip().lower() == "yes" and ground_truth_lower in response_lower



def build_messages(item):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    dialogue_turns = item["dialogue_turns"]
    q = item["test_query"] + "\n\n无需解释，直接回答。"
    for turn in dialogue_turns:
        user_input = turn["user_input"]
        llm_output = turn["llm_output"]
        messages.append({"role": "user", "content": user_input})
        messages.append({"role": "assistant", "content": llm_output})
    messages.append({"role": "user", "content": q})
    return messages


def build_messages_context(item):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    dialogue_turns = item["dialogue_turns"]
    q = item["test_query"] + "\n\n无需解释，直接回答。"
    history = []
    for turn in dialogue_turns:
        user_input = turn["user_input"]
        llm_output = turn["llm_output"]
        history.append(f"User: {user_input}\nAssistant: {llm_output}")
    context = "\n".join(history)
    prompt = f"""请根据给定的对话历史，回答下方的问题。
## 对话历史
{context}

## 问题
User: {q}
"""
    messages.append({"role": "user", "content": prompt})
    return messages
