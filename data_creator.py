from typing import List, Dict
import json
import os
import datetime
import asyncio
import uuid
import sys
from pathlib import Path


from loguru import logger
from tqdm.asyncio import tqdm as async_tqdm
import pnlp


sys.path.append("/backup/lanzhenzhongLab/haoshaochun/server_llm")

from llm_local import call_llm_local_async
from llm_async import call_llm_async

# 9643 tokens
prompt = pnlp.read_file("prompt_simple.txt")



async def do_llm_call(
    model: str, 
    temperature: float,
    msgs: List[Dict], 
    typ: str, 
    max_new_tokens: int = 2048, 
    return_reasoning: bool = False
):
    if model in [
        "gpt-oss-120b",
    ]:
        func = call_llm_local_async
    else:
        func = call_llm_async
    uid = str(uuid.uuid4())
    gen = func(
        model,
        msgs,
        f"dialogue_id_{typ}_{uid}",
        f"request_id_{typ}_{uid}",
        max_new_tokens,
        temperature,
        return_reasoning=return_reasoning,
    )
    res = ""
    r_res = ""
    async for v in gen:
        if v.endswith("<reasoning>"):
            r_res += v.strip("<reasoning>")
            continue
        if v.startswith(">>>FROM"):
            logger.info(f"Received chunk: {v}")
            break
        res += v
    return res, r_res


def json_satisfied(s):
    s = s.replace("```json", "").replace("```JSON", "").replace("```", "").strip()
    try:
        js = json.loads(s)
    except Exception as e:
        logger.error(f"JSON parse error: {e} with input: {s!r}")
        return False, None
    return True, js


def check_valid(dct: Dict):
    if isinstance(dct, List):
        dct = dct[0]
    if not dct:
        return False
    if not all((
        "dialogue_id" in dct,
        "domain" in dct,
        "key_facts" in dct,
        "dialogue_turns" in dct,
    )):
        logger.error(f"Missing top-level keys in {dct.keys()}")
        return False
    turns = dct["dialogue_turns"]
    for turn in turns:
        if not all((
            "turn_id" in turn,
            "user_input" in turn,
            "turn_type" in turn,
            "conflict_label_Lp" in turn,
            "authoritative_state_M_t" in turn,
            "ground_truth_response_LM" in turn,
            "winning_fact_id" in turn,
        )):
            logger.error(f"Missing turn-level keys in {turn.keys()}")
            return False
    return True



def check_valid_simple(dct: Dict):
    if isinstance(dct, List):
        dct = dct[0]
    if not dct or not isinstance(dct, Dict):
        return False
    if not all((
        "dialogue_turns" in dct,
        "test_query" in dct,
        "ground_truth" in dct,
    )):
        logger.error(f"Missing top-level keys in {dct.keys()}")
        return False
    turns = dct["dialogue_turns"]
    for turn in turns:
        if not all((
            "turn_id" in turn,
            "user_input" in turn,
            "authoritative_state_M_t" in turn,
            "llm_output" in turn,
        )):
            logger.error(f"Missing turn-level keys in {turn.keys()}")
            return False
    return True


async def process_item(i: str, model: str, temperature: float, cache: dict, semaphore):
    n_max_try = 2
    async with semaphore:
        if i in cache:
            return
        output = ""
        reasoning = ""
        count = 0
        msgs = [
            {"role": "system", "content": prompt},
        ]

        is_ok, parsed_data = json_satisfied(output)
        while not is_ok and count < n_max_try:
            count += 1
            try:
                output, reasoning = await do_llm_call(model, temperature, msgs, "generator", 20480, False)
            except Exception as e:
                continue
            logger.info(f"{output!r}")
            is_ok, parsed_data = json_satisfied(output)
            if not check_valid_simple(parsed_data):
                is_ok = False

        if is_ok:
            cache[i] = parsed_data
        else:
            logger.error(f"Failed to parse JSON after {n_max_try} tries: {output!r}")


async def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "gpt-oss-120b"
    temperature = 1.0

    dt = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("./data_simple/")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{model.replace('/', '_')}_{temperature}"
    out_file = f"data_{tag}_{dt}.jsonl"
    out_fp = out_dir / out_file

    cache_file = out_dir / f"cache_{tag}.json"
    if os.path.exists(cache_file):
        cache = pnlp.read_json(cache_file)
    else:
        cache = {}

    N_PROC = 1
    N_DATA = 1
    N_BATCH = N_DATA // N_PROC
    semaphore = asyncio.Semaphore(N_PROC)
    for batchi, batch in enumerate(range(N_BATCH)):
        tasks = [
            asyncio.create_task(
                process_item(str(i), model, temperature, cache, semaphore)
            ) for i in range(batchi * N_PROC, (batchi + 1) * N_PROC)
        ]
        for j, f in enumerate(async_tqdm(asyncio.as_completed(tasks), total=len(tasks)), start=1):
            await f

        pnlp.write_json(cache_file, cache, indent=2, ensure_ascii=False)
        logger.info(f"Batch {batchi} completed")

    out = list(cache.values())
    pnlp.write_list_dict_to_file(out_fp, out)
    if os.path.exists(cache_file):
        os.remove(cache_file)
    logger.info(f"Saved to {out_fp}")




if __name__ == "__main__":
    asyncio.run(main())
    # python data_creator.py openrouter/google/gemini-2.5-pro