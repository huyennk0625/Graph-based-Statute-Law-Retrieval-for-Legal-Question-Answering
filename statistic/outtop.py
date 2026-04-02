import json

TOP_K = 150

"""
statistic case out top
analysis "理由" 
"""

with open("coliee25/outputs/runs/bge-biencoder/en_25_train_bi_encoder_bge_m3.json", "r", encoding="utf-8") as f:
    train_bge = json.load(f)

with open("coliee25/outputs/runs/bge-biencoder/en_25_dev_bi_encoder_bge_m3.json", "r", encoding="utf-8") as f:
    dev_bge = json.load(f)

all_queries = train_bge["queries"] + dev_bge["queries"]

with open("coliee25/data/combine_train.json", "r", encoding="utf-8") as f:
    gold_data = json.load(f)

bge_map = {}

for q in all_queries:
    query_id = q["query_id"]

    top_articles = [
        item["article_id"]
        for item in q["results"][:TOP_K]
    ]

    bge_map[query_id] = set(top_articles)

missed = []

for sample in gold_data:
    query_id = sample["id"]

    if query_id not in bge_map:
        continue

    retrieved = bge_map[query_id]

    for rel in sample["relevant_articles"]:
        article_id = rel["article_id"]

        if article_id not in retrieved:
            missed.append({
                "query_id": query_id,
                "article_id": article_id
            })

with open("anew/missed_top150.json", "w", encoding="utf-8") as f:
    json.dump(missed, f, indent=2, ensure_ascii=False)

print(f"Total missed pairs: {len(missed)}")