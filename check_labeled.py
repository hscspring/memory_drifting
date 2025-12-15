import sys
import pnlp
from tqdm import tqdm

file = sys.argv[1]
data = pnlp.read_file_to_list_dict(file)
for i, v in enumerate(tqdm(data)):
    gt = v["ground_truth"]
    print(i+1, gt)
