from loguru import logger

from data_creator import do_llm_call


prompt_complex_reasoning = """You are a helpful and careful assistant specialized in multi-turn task-oriented dialogues.
Your goal is to help the user while accurately tracking all facts provided in the conversation history.

Instructions:
1. When answering a user question, first reason step by step.
2. Pay careful attention to all statements and facts in the conversation history.
3. Identify any updates, modifications, or conflicting information provided by the user across multiple turns.
4. Maintain a current, consistent representation of the user's facts (e.g., addresses, contact information, preferences).
5. If you detect conflicting or outdated facts, prioritize the most recent user-provided information, and explicitly note any changes.
6. Clearly indicate which facts you are using to answer the current query.
7. Provide answers that are accurate, concise, and grounded in the conversation history.

Return your output in a structured format that separates:
- Current Answer
- Facts Used
- Notes on any conflicts or updates

Use this structure consistently for each user query.
"""


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


def build_message_with_reasoning_prompt(item):
    prompt = """You are a helpful assistant. 
When answering the user's question, first think step by step and pay attention to the statements/facts made in the conversation history."""
    prompt = prompt_complex_reasoning
    return _build_messages(item, prompt)


def build_message_with_prompt(item):
    prompt = """You are a helpful assistant. 
When answering the user's question, pay attention to the statements/facts made in the conversation history."""
    return _build_messages(item, prompt)


def build_messages(item):
    return _build_messages(item, "You are a helpful assistant.")


def _build_messages(item, system_prompt):
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    dialogue_turns = item["dialogue_turns"]
    q = item["test_query"] + "\n\n无需解释，直接回答。"
    q = "当前状态是什么？\n\n无需解释，直接回答。"
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
    # q = "当前状态是什么？\n\n无需解释，直接回答。"
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