# README


```bash
# 数据生成
python data_creator.py openrouter/google/gemini-2.5-pro

# 校验重复性+合并
python check_and_combine_ds.py
# 校验最终label
python check_labeled.py

# 评测
python eval_local.py
python eval_close.py

# 先生成再评测
python eval_offline.py
```