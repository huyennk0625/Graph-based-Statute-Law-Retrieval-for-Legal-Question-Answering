import torch

from graphs.build_graph import build_edge_index, build_article_mapping
from retrieval.base import load_articles

def prepare_graph(edges_path, article_path):
    # load articles
    a_ids, articles = load_articles(article_path, lang="en")

    # mapping
    article2id, id2article = build_article_mapping(a_ids)

    # edge index
    edge_index = build_edge_index(edges_path, article2id)

    print("Num articles:", len(a_ids))
    print("Num edges:", edge_index.size(1))

    return a_ids, articles, article2id, id2article, edge_index

def build_training_pairs(label_file):
    import json
    pairs = []

    with open(label_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        q = item["query"]
        for art in item["relevant_articles"]:
            pairs.append((q, art["article_id"]))

    return pairs

def build_training_pairs_graphsage(label_file):
    import json
    pairs = []

    with open(label_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        qid = item["id"]
        q_text = item["query"]

        for art in item["relevant_articles"]:
            pairs.append((qid, q_text, art["article_id"]))

    return pairs