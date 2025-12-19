from datasets import load_dataset


data_path = "../mocker/kv_mock_2000.jsonl"
ds = load_dataset("json", data_files=data_path, split="train")
print(ds)
for v in ds:
    print(v)
    break
    