import sys
import copy
import pnlp
from tqdm import tqdm
from loguru import logger


from utils import check_answer, build_messages, build_messages_context
from data_creator import do_llm_call


model = sys.argv[1]
model_tag = model.split("/")[-1]



async def run_eval(item):
    msgs = build_messages_context(item)
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
    file = "data_simple/eval_ds_human.jsonl"
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