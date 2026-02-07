import json
from collections import defaultdict

TOP_KS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100, 200, 500]


def load_groundtruth(path):
    """
    Returns:
        gt_map: dict {query_id -> set(article_id)}
    """
    with open(path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    gt_map = {}
    for item in gt_data:
        qid = item["id"]
        relevant = {a["article_id"] for a in item["relevant_articles"]}
        gt_map[qid] = relevant

    return gt_map


def load_results(path):
    """
    Returns:
        results_map: dict {query_id -> list(article_id ranked)}
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results_map = {}
    for q in data["queries"]:
        qid = q["query_id"]
        ranked_articles = [r["article_id"] for r in q["results"]]
        results_map[qid] = ranked_articles

    return results_map

def evaluate(results_map, gt_map):
    metrics = {}

    for k in TOP_KS:
        precision_sum = 0.0
        recall_sum = 0.0
        f2_sum = 0.0
        valid_queries = 0

        for qid, relevant_articles in gt_map.items():
            retrieved = results_map.get(qid, [])[:k]
            retrieved_set = set(retrieved)

            num_correct = len(retrieved_set & relevant_articles)

            # per-query precision & recall
            precision_q = (
                num_correct / len(retrieved)
                if len(retrieved) > 0
                else 0.0
            )

            recall_q = (
                num_correct / len(relevant_articles)
                if len(relevant_articles) > 0
                else 0.0
            )

            if (4 * precision_q + recall_q) > 0:
                f2_q = (5 * precision_q * recall_q) / (
                    4 * precision_q + recall_q
                )
            else:
                f2_q = 0.0

            precision_sum += precision_q
            recall_sum += recall_q
            f2_sum += f2_q
            valid_queries += 1

        metrics[k] = {
            "Precision": precision_sum / valid_queries,
            "Recall": recall_sum / valid_queries,
            "F2": f2_sum / valid_queries,
        }

    return metrics

if __name__ == "__main__":
    results_path = "coliee25/models/graphencoder/test_graphsage_results.json"
    #results_path = "outputs/runs/bm25/coliee-23/en_23_test_results.json"
    gt_path = "coliee25/data/split-data/test.json"

    results_map = load_results(results_path)
    gt_map = load_groundtruth(gt_path)

    metrics = evaluate(results_map, gt_map)

    for k, m in metrics.items():
        print(
            f"Top-{k:3d} | "
            f"P={m['Precision']:.4f} "
            f"R={m['Recall']:.4f} "
            f"F2={m['F2']:.4f}"
        )
    