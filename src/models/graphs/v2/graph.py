import torch
from sentence_transformers import SentenceTransformer
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import json
import random
import numpy as np
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate(predictions, ground_truth, k=5):
    total_p, total_r, total_f2 = 0, 0, 0
    n = len(predictions)

    for qid in predictions:
        pred = set(predictions[qid][:k])
        gt = set(ground_truth[qid])

        if len(gt) == 0:
            continue

        tp = len(pred & gt)

        precision = tp / len(pred) if len(pred) > 0 else 0
        recall = tp / len(gt)

        if precision + recall == 0:
            f2 = 0
        else:
            f2 = (5 * precision * recall) / (4 * precision + recall)

        total_p += precision
        total_r += recall
        total_f2 += f2

    return total_p / n, total_r / n, total_f2 / n

def save_results_json(output_path, queries, a_ids, scores, topk_indices, method="Graph+BGE"):
    results = []

    for i, q in enumerate(queries):
        qid = q["id"]
        qtext = q["text"]

        query_results = []

        for idx in topk_indices[i]:
            idx = idx.item()
            query_results.append({
                "article_id": a_ids[idx],
                "score": float(scores[i, idx].item())
            })

        results.append({
            "query_id": qid,
            "query": qtext,
            "results": query_results
        })

    output = {
        "method": method,
        "num_queries": len(queries),
        "queries": results
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Saved results to {output_path}")

def load_train(train_path, corpus_path):
    with open(train_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus = json.load(f)

    queries = []

    for item in data:
        queries.append({
            'id': item['id'],
            'text': item['query'],
            'relevant_articles': item['relevant_articles']
        })

    return queries, corpus

def load_edges(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def build_article_text(corpus):
    articles = {}
    for item in corpus:
        aid = item["article_id"]
        text = item["article_content_en"]

        if "meta" in item and item["meta"]:
            text = item["meta"] + " " + text

        articles[aid] = text
    return articles

def build_article_edges(internal_edges, external_edges, aid2idx):
    src, dst = [], []

    def add_edges(edge_dict):
        for a, neighbors in edge_dict.items():
            if a not in aid2idx:
                continue
            for b in neighbors:
                if b in aid2idx:
                    src.append(aid2idx[a])
                    dst.append(aid2idx[b])

    add_edges(internal_edges)
    add_edges(external_edges)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return edge_index

model_bge = SentenceTransformer("coliee25/outputs/runs/testok66.67/en-epoch2")

def encode_articles(model, articles):
    a_ids = list(articles.keys())
    texts = [articles[aid] for aid in a_ids]

    emb = model.encode(
        texts,
        batch_size=32,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    return emb, a_ids


def encode_queries(model, queries):
    q_texts = [
        f"Represent this sentence for searching relevant passages: {q['text']}"
        for q in queries
    ]

    emb = model.encode(
        q_texts,
        batch_size=32,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    return emb

class ArticleGraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)

        self.alpha = nn.Parameter(torch.tensor(0.2))  # learnable weight

    def forward(self, x, edge_index):
        x1 = self.conv1(x, edge_index)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=0.2, training=self.training)
        x2 = self.conv2(x1, edge_index)

        return x + self.alpha * x2
    
def build_pos_pairs(train_data, qid2idx, aid2idx):
    pairs = []

    for item in train_data:
        qid = item["id"]
        for art in item["relevant_articles"]:
            aid = art["article_id"]
            if aid in aid2idx:
                pairs.append((qid2idx[qid], aid2idx[aid]))

    return pairs

def contrastive_loss(query_emb, article_emb, pos_pairs, temperature=0.05):
    query_emb = F.normalize(query_emb, dim=1)
    article_emb = F.normalize(article_emb, dim=1)

    logits = query_emb @ article_emb.T / temperature

    labels = torch.zeros_like(logits)

    for q, a in pos_pairs:
        labels[q, a] = 1

    log_prob = F.log_softmax(logits, dim=1)

    loss = -(labels * log_prob).sum(dim=1) / labels.sum(dim=1).clamp(min=1)

    return loss.mean()

def train():

    queries, corpus = load_train("coliee25/data/split-data/train.json", "coliee25/data/civil_code_parsed.json")
    articles = build_article_text(corpus)

    # encode
    article_emb_init, a_ids = encode_articles(model_bge, articles)
    query_emb = encode_queries(model_bge, queries)
    article_emb_init = article_emb_init.clone().detach().to(DEVICE)
    query_emb = query_emb.clone().detach().to(DEVICE)

    # mapping
    aid2idx = {aid: i for i, aid in enumerate(a_ids)}
    qid2idx = {q["id"]: i for i, q in enumerate(queries)}

    # graph
    external = load_edges("coliee25/data/graph/legal-graph-2025-external.json")
    internal = load_edges("coliee25/data/graph/legal-graph-2025-internal-w4.json")

    edge_index = build_article_edges(internal, external, aid2idx).to(DEVICE)


    model = ArticleGraphSAGE(
        in_dim=article_emb_init.shape[1],
        hidden_dim=256,
        out_dim=article_emb_init.shape[1]
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    pos_pairs = build_pos_pairs(queries, qid2idx, aid2idx)

    for epoch in range(6):
        model.train()
        optimizer.zero_grad()

        article_emb_graph = model(article_emb_init.to(DEVICE), edge_index)

        loss = contrastive_loss(
            query_emb.to(DEVICE),
            article_emb_graph,
            pos_pairs
        )

        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch} | Loss {loss.item():.4f}")
    
    torch.save(model.state_dict(), "coliee25/models/graphencoder/graphsage_model.pt")

def infer_and_evaluate(test_path, corpus_path):
    queries, corpus = load_train(test_path, corpus_path)
    articles = build_article_text(corpus)

    # encode
    article_emb_init, a_ids = encode_articles(model_bge, articles)
    query_emb = encode_queries(model_bge, queries)

    article_emb_init = article_emb_init.to(DEVICE)
    query_emb = query_emb.to(DEVICE)

    # mapping
    aid2idx = {aid: i for i, aid in enumerate(a_ids)}

    # graph
    external = load_edges("coliee25/data/graph/legal-graph-2025-external.json")
    internal = load_edges("coliee25/data/graph/legal-graph-2025-internal-w4.json")
    edge_index = build_article_edges(internal, external, aid2idx).to(DEVICE)

    # load model
    model = ArticleGraphSAGE(
        in_dim=article_emb_init.shape[1],
        hidden_dim=256,
        out_dim=article_emb_init.shape[1]
    ).to(DEVICE)

    model.load_state_dict(torch.load("coliee25/models/graphencoder/graphsage_model.pt"))
    model.eval()

    # forward
    with torch.no_grad():
        article_emb = model(article_emb_init, edge_index)

        query_emb = F.normalize(query_emb, dim=1)
        article_emb = F.normalize(article_emb, dim=1)

        scores = query_emb @ article_emb.T

    # ranking
    """top_k = 10
    topk_indices = scores.topk(top_k, dim=1).indices"""
    top_k = 200
    topk = scores.topk(top_k, dim=1)
    topk_indices = topk.indices
    topk_scores = topk.values

    predictions = {}
    ground_truth = {}

    for i, q in enumerate(queries):
        qid = q["id"]

        pred_articles = [a_ids[idx] for idx in topk_indices[i].cpu().numpy()]
        predictions[qid] = pred_articles

        gt_articles = [art["article_id"] for art in q["relevant_articles"]]
        ground_truth[qid] = gt_articles

    # evaluate
    for k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100, 200]:
        p, r, f2 = evaluate(predictions, ground_truth, k=k)
        print(
            f"P@{k}: {p:.4f}| "
            f"R@{k}: {r:.4f}| "
            f"F2@{k}: {f2:.4f}| "
        )
    
    save_results_json(
        output_path="coliee25/models/graphencoder/test_graphsage_results.json",
        queries=queries,
        a_ids=a_ids,
        scores=scores,
        topk_indices=topk_indices,
        method="GraphSAGE+BGE"
    )

    return predictions

if __name__ == "__main__":
    set_seed(42)
    train()

    preds = infer_and_evaluate(
        test_path="coliee25/data/split-data/test.json",
        corpus_path="coliee25/data/civil_code_parsed.json"
    )