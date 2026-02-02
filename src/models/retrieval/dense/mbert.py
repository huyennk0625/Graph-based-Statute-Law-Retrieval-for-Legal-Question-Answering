import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List
from retrieval.base import load_queries, load_articles, save_results

"""
code bert không fine-tune, dùng mean pooling
cho kết quả thấp hơn bm25
"""

class MBertRetriever:
    def __init__(
        self,
        model_name: str = "google-bert/bert-base-multilingual-cased",
        device: str = None,
        batch_size: int = 16,
        max_length: int = 512
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.doc_ids = None
        self.doc_embeddings = None

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return (token_embeddings * mask).sum(1) / mask.sum(1)

    @torch.no_grad()
    def _encode(self, texts: List[str]) -> torch.Tensor:
        embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)

            outputs = self.model(**inputs)
            emb = self._mean_pooling(outputs, inputs["attention_mask"])
            emb = F.normalize(emb, p=2, dim=1)  # cosine similarity

            embeddings.append(emb.cpu())

        return torch.cat(embeddings, dim=0)

    def fit(self, doc_ids: List[str], doc_texts: List[str]):
        """
        Encode and store document embeddings
        """
        self.doc_ids = doc_ids
        self.doc_embeddings = self._encode(doc_texts)

    def search(self, query_texts: List[str], top_k: int = 10):
        """
        Return: List[List[(doc_id, score)]]
        """
        if self.doc_embeddings is None:
            raise ValueError("You must call fit() before search().")

        query_embeddings = self._encode(query_texts)

        scores = torch.matmul(query_embeddings, self.doc_embeddings.T)

        topk_scores, topk_indices = torch.topk(scores, k=top_k, dim=1)

        all_results = []
        for i in range(len(query_texts)):
            results = []
            for score, idx in zip(topk_scores[i], topk_indices[i]):
                results.append(
                    (self.doc_ids[idx], float(score))
                )
            all_results.append(results)

        return all_results

lang = "jp"
query_ids, query_texts = load_queries("data/coliee-2023/data-preprocess/combined_en_jp_train_data.json", lang)
doc_ids, docs = load_articles("data/coliee-2023/data-preprocess/stage-2-corpus.json", lang)

retriever = MBertRetriever(batch_size=16)
retriever.fit(doc_ids, docs)
results = retriever.search(query_texts, top_k=700)

save_results(
    path="outputs/runs/mbert/jp_23_train_results.json",
    query_ids=query_ids,
    queries=query_texts,
    all_results=results,
    method="bi-encoder-mbert"
)
