import json
import random

""""
    Split combine train -> train + dev
    dev = 2 test (73x2=146)
    Shuffle(42) before split 
"""

INPUT_PATH = "coliee25/data/combine_train.json"
TRAIN_OUT = "coliee25/data/split-data/train.json"
DEV_OUT = "coliee25/data/split-data/dev.json"
DEV_SIZE = 146
SEED = 42

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print("Total samples:", len(data))

random.seed(SEED)
random.shuffle(data)

dev_data = data[:DEV_SIZE]
train_data = data[DEV_SIZE:]

print("Train size:", len(train_data))
print("Dev size:", len(dev_data))

print(
    sum(len(x["relevant_articles"]) for x in dev_data) / len(dev_data)
)

print(
    sum(len(x["relevant_articles"]) for x in train_data) / len(train_data)
)

with open(TRAIN_OUT, "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=4)

with open(DEV_OUT, "w", encoding="utf-8") as f:
    json.dump(dev_data, f, ensure_ascii=False, indent=4)

print("Done")