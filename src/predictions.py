import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sentence_transformers import SentenceTransformer

from src.models.graphs.v2.graph import (
    set_seed, build_article_text,
    build_article_edges, load_edges,
    ArticleGraphSAGE
)

from src.models.retrieval.lexical.bm25 import (
    BM25Retriever
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class HybridRetriever:

    def __init__(
        self,
        corpus_path,
        graph_model_path,
        internal_edge_path,
        external_edge_path,
        bge_model_path,
        lang="en",
        bm25_topk=700,
        graph_topk=200,
        alpha=0.15,
        threshold=0.88
    ):

        self.lang = lang

        self.bm25_topk = bm25_topk
        self.graph_topk = graph_topk

        self.alpha = alpha
        self.threshold = threshold

        corpus = load_json(corpus_path)

        self.articles = build_article_text(corpus)

        self.a_ids = list(self.articles.keys())

        self.aid2idx = {
            aid: i
            for i, aid in enumerate(self.a_ids)
        }

        self.bm25 = BM25Retriever(lang=lang)

        self.bm25.fit(
            self.a_ids,
            list(self.articles.values())
        )

        self.model_bge = SentenceTransformer(
            bge_model_path
        )

        print("Encoding article embeddings...")

        self.article_emb_init = self.model_bge.encode(
            list(self.articles.values()),
            batch_size=32,
            convert_to_tensor=True,
            normalize_embeddings=True
        ).to(DEVICE)

        internal = load_edges(internal_edge_path)

        external = load_edges(external_edge_path)

        self.edge_index = build_article_edges(
            internal,
            external,
            self.aid2idx
        ).to(DEVICE)

        self.graph_model = ArticleGraphSAGE(
            in_dim=self.article_emb_init.shape[1],
            hidden_dim=256,
            out_dim=self.article_emb_init.shape[1]
        ).to(DEVICE)

        self.graph_model.load_state_dict(
            torch.load(graph_model_path)
        )

        self.graph_model.eval()

        print("Building graph-enhanced embeddings...")

        with torch.no_grad():

            self.article_emb_graph = self.graph_model(
                self.article_emb_init,
                self.edge_index
            )

            self.article_emb_graph = F.normalize(
                self.article_emb_graph,
                dim=1
            )

        print("Retriever ready.")


    def normalize_bm25(self, bm25_results):

        if len(bm25_results) == 0:
            return {}

        scores = [score for _, score in bm25_results]

        max_score = max(scores)
        min_score = min(scores)

        norm_scores = {}

        for aid, score in bm25_results:

            if max_score == min_score:
                norm = 0.0

            else:
                norm = (score - min_score) / (
                    max_score - min_score
                )

            norm_scores[aid] = norm

        return norm_scores

    def retrieve(self, query, top_k=10):

        bm25_results = self.bm25.search(
            query,
            top_k=self.bm25_topk
        )

        bm25_scores = self.normalize_bm25(
            bm25_results
        )

        q_emb = self.model_bge.encode(
            [
                f"Represent this sentence for searching relevant passages: {query}"
            ],
            convert_to_tensor=True,
            normalize_embeddings=True
        ).to(DEVICE)

        with torch.no_grad():

            graph_scores = (
                q_emb @ self.article_emb_graph.T
            ).squeeze(0)

        final_scores = {}

        for idx, aid in enumerate(self.a_ids):

            bm25_score = bm25_scores.get(aid, 0.0)

            graph_score = graph_scores[idx].item()

            final_score = (
                self.alpha * bm25_score
                + (1 - self.alpha) * graph_score
            )

            if final_score >= self.threshold:

                final_scores[aid] = {
                    "final_score": final_score,
                    "bm25_score": bm25_score,
                    "graph_score": graph_score
                }

        ranked = sorted(
            final_scores.items(),
            key=lambda x: x[1]["final_score"],
            reverse=True
        )[:top_k]

        return ranked


if __name__ == "__main__":

    set_seed(42)

    retriever = HybridRetriever(
        corpus_path="coliee25/data/civil_code_parsed.json",

        graph_model_path=(
            "coliee25/models/graphencoder/"
            "graphsage_model.pt"
        ),

        internal_edge_path=(
            "coliee25/data/graph/"
            "legal-graph-2025-internal-w4.json"
        ),

        external_edge_path=(
            "coliee25/data/graph/"
            "legal-graph-2025-external.json"
        ),

        bge_model_path=(
            "coliee25/outputs/runs/"
            "testok66.67/en-epoch2"
        ),

        alpha=0.15,
        threshold=0.5
    )

    """
    When a petition for the commencement of curatorship is filed for a person who constantly lacks the capacity to appreciate a given situation due to mental disabilities, the family court may decide to establish a curatorship issue

    If a person under adult guardianship purchases daily necessities without the consent of the adult guardian, the adult guardian may cancel the contract involving the purchase.
      """  

    while True:
      query = input("\nEnter query (type 'exit' to quit):\n")

      if query.lower() == "exit":
        break

      results = retriever.retrieve(
          query,
          top_k=3
      )

      print("\nQuery:")
      print(query.strip())

      for rank, (aid, info) in enumerate(results, start=1):

          print(f"\\nRank {rank}")
          print(f"Article ID : {aid}")

          print("\nArticle Text:")
          print(retriever.articles[aid])