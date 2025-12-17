from pathlib import Path
from collections import defaultdict

import pnlp

root = "./data_simple/multi-turn"
data_lines = pnlp.read_file_to_list_dict("data_simple/eval_ds_human.jsonl")
error_dict = defaultdict(list)
n_models = 0
all_models = []
for file in Path(root).glob("*.jsonl"):
    lines = pnlp.read_file_to_list_dict(file)
    model = file.stem.replace("eval_ds_human_eval_", "")
    all_models.append(model)
    n_models += 1
    for i, js in enumerate(lines):
        if not js["is_correct"]:
            answer = js["response"]
            error_dict[i].append(model + "->" + answer)
            

sorted_res = sorted(error_dict.items(), key=lambda x: -len(x[1]))
for idx, models in sorted_res:
    data_lines[idx]["error_models"] = models

for idx, v in enumerate(data_lines):
    if "error_models" not in v:
        v["error_models"] = []
    v["index"] = idx
sorted_data = sorted(data_lines, key=lambda x: -len(x["error_models"]))
out_res = []
for item in sorted_data:
    new = {}
    error_models = item["error_models"]
    error_count = len(error_models)
    new["error_model_count"] = f"{error_count}/{n_models}"
    new["error_models"] = error_models
    _error_models = [v.split("->")[0] for v in error_models]
    new["correct_models"] = [k for k in all_models if k not in _error_models]
    for key in item:
        if key not in new:
            new[key] = item[key]
    out_res.append(new)

pnlp.write_list_dict_to_file("multiturn.jsonl", out_res)