import json
from collections import defaultdict

BGE_SCORE_PATH = "coliee25/outputs/runs/bge-biencoder/en_25_test_bi_encoder_bge_m3.json"
EDGE_PATH = "coliee25/data/graph/legal-graph-2025-external.json"
OUTPUT_PATH = "anew/test_bge_graph_rerank.json"

TOP_K = 100
ALPHA = 0.02

with open(BGE_SCORE_PATH, "r", encoding="utf-8") as f:
    bge_data = json.load(f)

with open(EDGE_PATH, "r", encoding="utf-8") as f:
    edges = json.load(f)

incoming_graph = defaultdict(list)

for src, dst_list in edges.items():
    for dst in dst_list:
        incoming_graph[dst].append(src)

new_queries = []

for query_obj in bge_data["queries"]:

    top_results = query_obj["results"][:TOP_K]

    score_dict = {
        item["article_id"]: item["score"]
        for item in top_results
    }

    candidate_articles = set(score_dict.keys())

    reranked = []

    for article_id in candidate_articles:

        original_score = score_dict.get(article_id, 0.0)

        propagated_score = 0.0

        incoming_nodes = incoming_graph.get(article_id, [])

        for src in incoming_nodes:

            if src in score_dict:
                out_degree = max(len(edges.get(src, [])), 1)

                propagated_score += score_dict[src] / out_degree

        new_score = original_score + ALPHA * propagated_score

        reranked.append({
            "article_id": article_id,
            "score": float(new_score),
            "original_score": float(original_score),
            "propagated_score": float(propagated_score)
        })

    reranked.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    new_queries.append({
        "query_id": query_obj["query_id"],
        "query": query_obj["query"],
        "results": reranked
    })

output = {
    "method": f"bge-graph-rerank-alpha-{ALPHA}",
    "top_k": TOP_K,
    "num_queries": len(new_queries),
    "queries": new_queries
}

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"Saved to {OUTPUT_PATH}")