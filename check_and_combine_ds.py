from pathlib import Path

import pnlp


root = Path("./data_simple")
out_fp = root / "eval_ds.jsonl"
out = []
cache = {}
for file in root.glob("data_*.jsonl"):
    data = pnlp.read_file_to_list_dict(file)
    for v in data:
        key = ""
        for turn in v["dialogue_turns"]:
            inp = turn["user_input"]
            oup = turn["llm_output"]
            key += inp + oup
        if key in cache:
            continue
        cache[key] = v
        out.append(v)
print(len(out))
pnlp.write_list_dict_to_file(out_fp, out)