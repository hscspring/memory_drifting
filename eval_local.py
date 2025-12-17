import sys
import copy
import pnlp
from tqdm import tqdm
from loguru import logger
from transformers import AutoTokenizer, AutoModelForCausalLM


from utils import check_answer, build_messages, build_messages_context, build_message_with_reasoning_prompt, build_message_with_prompt


# model_id = "/backup/lanzhenzhongLab/public/models/Qwen2.5-7B-Instruct"
# model_id = "/backup/lanzhenzhongLab/public/models/Qwen2.5-14B-Instruct"
# model_id = "/backup/lanzhenzhongLab/public/models/Qwen3-4B-Instruct-2507"
model_id = "/backup/lanzhenzhongLab/public/models/Qwen3-8B"

model_tag = model_id.split("/")[-1]
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto"
)

def build_inputs(item, enable_thinking: bool = False):
    messages = build_message_with_reasoning_prompt(item)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    return model_inputs


async def run_eval(item, enable_thinking: bool = False):
    model_inputs = build_inputs(item, enable_thinking)
    gt = item["ground_truth"]
    if enable_thinking:
        max_new_tokens = 1024
    else:
        max_new_tokens = 128
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    thinking_content = ""
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    logger.info(f"Model response: {response}")
    if enable_thinking:
        tmp = response.split("</think>")
        if len(tmp) != 2:
            logger.warning(f"Unexpected thinking format: {response}")
        else:
            thinking_content = tmp[0]
            response = tmp[-1].strip()

    is_correct = await check_answer(response, gt)
    res = {
        "is_correct": is_correct,
        "response": response,
        "ground_truth": gt,
        "thinking": thinking_content,
    }
    return res


async def main():
    file = "data_simple/eval_ds_human.jsonl"
    ds = pnlp.read_file_to_list_dict(file)
    results = []
    for item in tqdm(ds):
        result = await run_eval(item, True)
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
    # CUDA_VISIBLE_DEVICES=7 python eval_local.py 