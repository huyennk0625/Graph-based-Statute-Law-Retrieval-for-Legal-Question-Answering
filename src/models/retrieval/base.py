import json
from typing import List

"""
load queries and load articles from data-preprocess
queries from combined data en_jp
articles from stage-2-corpus

    {
        "id": "R04-01-I",
        "yn_label": "N",
        "query": "A juridical ...",
        "relevant_articles": [
            {
                "article_id": "Article ...",
                "article_content": "..."
            },
            {
                "article_id": "Article ...",
                "article_content": "..."
            }
        ],
        "jp_query": "営業"
    },
"""

def load_queries(path: str, lang: str = "en"):
    with open(path, encoding="utf-8") as file:
        queries = json.load(file)
    if lang == "en":
        text = [q["query"] for q in queries]
    else:
        text = [q["jp_query"] for q in queries]
    ids = [q["id"] for q in queries]
    return ids, text

def load_articles(path: str, lang: str = "en"):
    with open(path, encoding="utf-8") as f:
        articles = json.load(f)
    if lang == "en":
        texts = [a["article_content_en"] for a in articles]
    else:
        texts = [a["article_content_jp"] for a in articles]
    ids = [a["article_id"] for a in articles]
    return ids, texts

def tokenize_en(text: str) -> List[str]:
    return text.lower().split()

def tokenize_jp(text: str, tokenizer) -> List[str]:
    return [m.surface() for m in tokenizer.tokenize(text)]

def save_results(path: str, query_ids: list, queries: List[str], all_results: List[list], method: str):
    """
    queries: List of query strings
    all_results: List of retrieval results, each = [(doc_id, score), ...]
    """
    output = {
    "method": method,
    "num_queries": len(queries),
    "queries": []
    }

    for qid, q, results in zip(query_ids, queries, all_results):
        output["queries"].append({
            "query_id": qid,
            "query": q,
            "results": [
                {
                    "article_id": doc_id,
                    "score": float(score)
                }
                for doc_id, score in results
            ]
        })


    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
