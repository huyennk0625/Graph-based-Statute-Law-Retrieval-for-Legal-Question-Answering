import numpy as np
import json
from metrics.metric import evaluate, load_groundtruth

def evaluate_threshold(results_map, gt_map):
    """
    results_map: {query_id -> set(article_id)}
    gt_map: {query_id -> set(relevant_article_id)}
    """

    precision_sum = 0.0
    recall_sum = 0.0
    f2_sum = 0.0
    valid_queries = 0

    for qid, relevant in gt_map.items():
        if not relevant:
            continue

        retrieved = results_map.get(qid, set())
        tp = len(retrieved & relevant)

        precision_q = tp / len(retrieved) if len(retrieved) > 0 else 0.0
        recall_q = tp / len(relevant)

        if (4 * precision_q + recall_q) > 0:
            f2_q = (5 * precision_q * recall_q) / (4 * precision_q + recall_q)
        else:
            f2_q = 0.0

        precision_sum += precision_q
        recall_sum += recall_q
        f2_sum += f2_q
        valid_queries += 1

    return {
        "Precision": precision_sum / valid_queries,
        "Recall": recall_sum / valid_queries,
        "F2": f2_sum / valid_queries
    }

def min_max_normalize(scores_dict, eps=1e-8):
    if not scores_dict:
        return {}

    values = list(scores_dict.values())
    min_s, max_s = min(values), max(values)

    if max_s - min_s < eps:
        return {k: 1.0 for k in scores_dict}

    return {
        k: (v - min_s) / (max_s - min_s)
        for k, v in scores_dict.items()
    }


def combine_per_query(bm25_results, mbert_results, a):
    b = 1.0 - a

    bm25 = {r["article_id"]: r["score"] for r in bm25_results}
    mbert = {r["article_id"]: r["score"] for r in mbert_results}

    all_docs = set(bm25) | set(mbert)

    bm25_full = {d: bm25.get(d, 0.0) for d in all_docs}
    mbert_full = {d: mbert.get(d, 0.0) for d in all_docs}

    bm25_norm = min_max_normalize(bm25_full)
    mbert_norm = min_max_normalize(mbert_full)

    combined = {
        d: a * bm25_norm[d] + b * mbert_norm[d]
        for d in all_docs
    }

    return combined  # {article_id: score}

def build_results_map_threshold_only(
    bm25_data,
    mbert_data,
    a,
    threshold
):
    """
    Return
        results_map: {query_id -> set(article_id)}
    """

    mbert_index = {
        q["query_id"]: q["results"]
        for q in mbert_data["queries"]
    }

    results_map = {}

    for q in bm25_data["queries"]:
        qid = q["query_id"]

        bm25_results = q["results"]
        mbert_results = mbert_index.get(qid, [])

        combined_scores = combine_per_query(
            bm25_results,
            mbert_results,
            a
        )

        retrieved = {
            doc_id
            for doc_id, score in combined_scores.items()
            if score >= threshold
        }

        results_map[qid] = retrieved

    return results_map

def grid_search_threshold_only(
    bm25_data,
    mbert_data,
    gt_map,
    output_path,
    alphas=np.linspace(0, 1, 11),
    thresholds=np.linspace(0, 1, 101)
):
    records = []

    for a in alphas:
        b = 1.0 - a

        for th in thresholds:
            results_map = build_results_map_threshold_only(
                bm25_data=bm25_data,
                mbert_data=mbert_data,
                a=a,
                threshold=float(th)
            )

            metrics = evaluate_threshold(results_map, gt_map)

            records.append({
                "a": round(float(a), 4),
                "b": round(float(b), 4),
                "threshold": round(float(th), 4),
                "precision": round(metrics["Precision"], 6),
                "recall": round(metrics["Recall"], 6),
                "f2": round(metrics["F2"], 6)
            })

        print(f"done a={a:.2f}")

    records.sort(key=lambda x: x["f2"], reverse=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    return records

if __name__ == "__main__":
    bm25_path = "outputs/runs/bm25/coliee-23-dev/jp_23_dev_results.json"
    mbert_path = "outputs/runs/mbert-biencoder/infer-dev/jp_dev_23_bi_encoder_mbert.json"
    gt_path = "data/coliee-2023/data-spliter/en_jp_dev_set.json"


    with open(bm25_path, "r", encoding="utf-8") as f:
        bm25_data = json.load(f)

    with open(mbert_path, "r", encoding="utf-8") as f:
        mbert_data = json.load(f)

    gt_map = load_groundtruth(gt_path)

    grid_search_threshold_only(
        bm25_data=bm25_data,
        mbert_data=mbert_data,
        gt_map=gt_map,
        output_path="test/bge-graph.json",
        alphas=np.linspace(0, 1, 21),
        thresholds=np.linspace(0, 1, 101)
    )
