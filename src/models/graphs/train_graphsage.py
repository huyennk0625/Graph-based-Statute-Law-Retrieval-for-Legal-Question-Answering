import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from graphsage_model import GraphSAGEModel
import random
import json

device = "cuda" if torch.cuda.is_available() else "cpu"
print("CUDA available:", torch.cuda.is_available())
print("Device:", device)

def encode_texts(model, texts, device):
    return model.encode(
        texts,
        convert_to_tensor=True,
        device=device,
        normalize_embeddings=True
    )

def load_training_triples(path):
    with open(path) as f:
        data = json.load(f)

    triples = []
    for item in data:
        triples.append((
            item["query"],
            item["positive"],
            item["negative"]
        ))
    return triples

def train_graphsage(
    model_path,
    articles,
    training_triples,
    article2id,
    edge_index,
    epochs=30
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sent_model = SentenceTransformer(model_path).to(device)

    print("Encoding articles...")
    doc_embs = encode_texts(sent_model, articles, device)

    print("Encoding queries...")
    queries = [
      f"Represent this sentence for searching relevant passages: {t[0]}"
      for t in training_triples
    ]
    query_embs = encode_texts(sent_model, queries, device)

    dim = doc_embs.shape[1]
    graphsage = GraphSAGEModel(dim).to(device)

    optimizer = torch.optim.Adam(graphsage.parameters(), lr=1e-4)
    edge_index = edge_index.to(device)

    print(edge_index.shape)
    print(edge_index[:, :10])
  

    for epoch in range(epochs):
        graphsage.train()

        x = doc_embs.clone()
        doc_embs_graph = graphsage(x, edge_index)
        doc_embs_graph = F.normalize(doc_embs_graph, dim=1)
        with torch.no_grad():
          cos = F.cosine_similarity(doc_embs, doc_embs_graph, dim=1)
          print(f"Epoch {epoch} | Cosine drift: {cos.mean().item():.4f}")

        triplet_losses = []
        #recon_losses = []

        indices = list(range(len(training_triples)))
        random.shuffle(indices)

        for i in indices:
          q_text, pos_article, neg_article = training_triples[i]
          q_emb = query_embs[i]

          pos_id = article2id[pos_article]
          neg_id = article2id[neg_article]

          pos_emb = doc_embs_graph[pos_id]
          neg_emb = doc_embs_graph[neg_id]

          pos_score = F.cosine_similarity(q_emb, pos_emb, dim=0)
          neg_score = F.cosine_similarity(q_emb, neg_emb, dim=0)

          margin = 0.2
          triplet_losses.append(F.relu(margin + neg_score - pos_score))

          # reconstruction loss
          target = doc_embs[pos_id].clone().detach()
          #recon_losses.append(F.mse_loss(doc_embs_graph[pos_id], target))
          #recon_losses.append(F.mse_loss(doc_embs_graph[pos_id], doc_embs[pos_id]))


        triplet_loss = torch.stack(triplet_losses).mean()
        recon_loss = 1 - F.cosine_similarity(doc_embs_graph, doc_embs, dim=1).mean()
        print(f"Triplet: {triplet_loss.item():.4f} | Recon: {recon_loss.item():.4f}")
        #recon_loss = torch.stack(recon_losses).mean()

        loss = triplet_loss + 0.1 * recon_loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"Epoch {epoch} Loss: {loss.item():.4f}")

    return graphsage