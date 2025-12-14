import sys
import copy
import pnlp
from tqdm import tqdm
from loguru import logger


from data_creator import do_llm_call

file = sys.argv[1]
model = sys.argv[2]

model_tag = model.split("/")[-1]


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
    output, reasoning = await do_llm_call("gpt-oss-120b", 0.7, msgs, "judger", 1024, False)
    logger.info(f"Judger output: {output}, reasoning: {reasoning}")
    return output.strip().lower() == "yes"


def build_inputs(item):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    dialogue_turns = item["dialogue_turns"]
    q = item["test_query"] + "\n\n无需解释，直接回答我。"
    for turn in dialogue_turns:
        user_input = turn["user_input"]
        llm_output = turn["llm_output"]
        messages.append({"role": "user", "content": user_input})
        messages.append({"role": "assistant", "content": llm_output})
    messages.append({"role": "user", "content": q})
    return messages


async def run_eval(item):
    msgs = build_inputs(item)
    gt = item["ground_truth"]
    output, reasoning = await do_llm_call(model, 0.7, msgs, "policy", 1024, False)
    response = output.strip()
    is_correct = await check_answer(response, gt)
    res = {
        "is_correct": is_correct,
        "response": response,
        "ground_truth": gt
    }
    return res


async def main():
    ds = pnlp.read_file_to_list_dict(file)
    results = []
    for item in tqdm(ds):
        result = await run_eval(item)
        results.append(result)

    n_correct = sum(1 for r in results if r["is_correct"])
    accuracy = n_correct / len(results)
    print(f"Accuracy: {accuracy:.4f} ({n_correct}/{len(results)})")
    out_file = file.replace(".jsonl", f"_eval_{model_tag}.jsonl")
    pnlp.write_list_dict_to_file(out_file, results)
    print(f"Saved to {out_file}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())