import json

"""
compare rank before and after user propagation
"""
ORIGINAL_PATH = "coliee25/outputs/runs/bge-biencoder/en_25_test_bi_encoder_bge_m3.json"
RERANK_PATH = "anew/test_bge_graph_rerank.json"
GOLD_PATH = "coliee25/data/split-data/test.json"

with open(ORIGINAL_PATH, "r", encoding="utf-8") as f:
    original_data = json.load(f)

with open(RERANK_PATH, "r", encoding="utf-8") as f:
    rerank_data = json.load(f)

with open(GOLD_PATH, "r", encoding="utf-8") as f:
    gold_data = json.load(f)

gold_map = {}

for item in gold_data:
    query_id = item["id"]

    relevant_articles = [
        art["article_id"]
        for art in item["relevant_articles"]
    ]

    gold_map[query_id] = relevant_articles

original_map = {
    q["query_id"]: q["results"]
    for q in original_data["queries"]
}

rerank_map = {
    q["query_id"]: q["results"]
    for q in rerank_data["queries"]
}

improved_cases = []

for query_id, gold_articles in gold_map.items():

    if query_id not in original_map:
        continue

    original_results = original_map[query_id]
    rerank_results = rerank_map[query_id]

    # article -> rank
    original_rank = {
        item["article_id"]: idx + 1
        for idx, item in enumerate(original_results)
    }

    rerank_rank = {
        item["article_id"]: idx + 1
        for idx, item in enumerate(rerank_results)
    }

    improved_articles = []

    for article_id in gold_articles:

        old_rank = original_rank.get(article_id, 999999)
        new_rank = rerank_rank.get(article_id, 999999)

        if new_rank < old_rank:

            improved_articles.append({
                "article_id": article_id,
                "old_rank": old_rank,
                "new_rank": new_rank,
                "improvement": old_rank - new_rank
            })

    if improved_articles:

        improved_cases.append({
            "query_id": query_id,
            "query": next(
                q["query"]
                for q in rerank_data["queries"]
                if q["query_id"] == query_id
            ),
            "improved_articles": improved_articles
        })

print(f"Total improved queries: {len(improved_cases)}")
print("=" * 80)

for case in improved_cases:

    print(f"\nQuery ID: {case['query_id']}")
    print(f"Query: {case['query']}")

    for art in case["improved_articles"]:

        print(
            f"  {art['article_id']}: "
            f"{art['old_rank']} -> {art['new_rank']} "
            f"(+{art['improvement']})"
        )