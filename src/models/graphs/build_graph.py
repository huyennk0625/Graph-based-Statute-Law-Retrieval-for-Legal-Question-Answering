import json
import torch

def build_edge_index(edges_json_path, article2id):
    with open(edges_json_path, "r", encoding="utf-8") as f:
        edges = json.load(f)

    edge_set = set()

    for src_article, neighbors in edges.items():
        if src_article not in article2id:
            continue
        src = article2id[src_article] # hiện tại

        for dst_article in neighbors:
            if dst_article not in article2id:
                continue

            dst = article2id[dst_article] # neighbor

            # add edge
            edge_set.add((dst, src)) # ref -> article
            #edge_set.add((src, dst))  # undirected graph

    edge_list = list(edge_set)
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    return edge_index

def build_article_mapping(article_ids):
    article2id = {aid: i for i, aid in enumerate(article_ids)}
    id2article = {i: aid for aid, i in article2id.items()}
    return article2id, id2article

def edge_index_to_dict(edge_index, id2article):
    graph = {}

    edges = edge_index.t().tolist()

    for src, dst in edges:
        src_id = id2article[src]
        dst_id = id2article[dst]

        graph.setdefault(src_id, set()).add(dst_id)

    return graph