import json
import torch
from sentence_transformers import SentenceTransformer, util
from typing import List, Tuple
from retrieval.base import load_queries, load_articles

def encode_queries_bge(model, queries, batch_size=32):
    queries = [
        f"Represent this sentence for searching relevant passages: {q}"
        for q in queries
    ]
    return model.encode(
        queries,
        batch_size=batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

def encode_docs_bge(model, docs, batch_size=32):
    return model.encode(
        docs,
        batch_size=batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True
    )


def retrieve_topk(
    query_embs,
    doc_embs,
    doc_ids: List[str],
    k: int = 10
):
    scores = util.cos_sim(query_embs, doc_embs)  # (n_query, n_doc)

    results = []
    for i in range(scores.size(0)):
        topk = torch.topk(scores[i], k=k)
        hits = [
            {
                "article_id": doc_ids[idx],
                "score": float(topk.values[j])
            }
            for j, idx in enumerate(topk.indices)
        ]
        results.append(hits)

    return results

def run_inference(
    model_path: str,
    query_path: str,
    article_path: str,
    lang: str = "en",
    top_k: int = 10
):
    print(f"Loading model: {model_path}")
    model = SentenceTransformer(model_path)

    print("Loading queries...")
    q_ids, queries = load_queries(query_path, lang)

    print("Loading articles...")
    a_ids, articles = load_articles(article_path, lang)

    print("Encoding articles...")
    doc_embs = encode_docs_bge(model, articles)

    print("Encoding queries...")
    query_embs = encode_queries_bge(model, queries)

    print("Retrieving...")
    results = retrieve_topk(
        query_embs,
        doc_embs,
        a_ids,
        k=top_k
    )

    return q_ids, results

import json

def save_results(
    output_path: str,
    method: str,
    query_ids,
    queries,
    results
):
    output = {
        "method": method,
        "num_queries": len(query_ids),
        "queries": []
    }

    for qid, qtext, res in zip(query_ids, queries, results):
        output["queries"].append({
            "query_id": qid,
            "query": qtext,
            "results": res
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":

    q_ids, queries = load_queries("coliee25/data/split-data/test.json", lang="en")

    q_ids, results = run_inference(
        model_path="./coliee25/outputs/runs/bge-triplet-final/model",
        query_path="coliee25/data/split-data/test.json",
        article_path="coliee25/data/civil_code_parsed.json",
        lang="en",
        top_k=500
    )

    save_results(
        output_path="./coliee25/outputs/runs/bge-triplet-final/infer/test_25_bge_m3.json",
        method="bi-encoder-bge-m3",
        query_ids=q_ids,
        queries=queries,
        results=results
    )