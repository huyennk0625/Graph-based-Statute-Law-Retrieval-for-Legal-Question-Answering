from sentence_transformers import SentenceTransformer
import numpy as np
from retrieval.base import load_queries, load_articles, save_results

"""
mpnet v2 không fine-tune
cho kết quả cao hơn mbert fine-tuned, nhỏ hơn bm25
"""

class BiEncoderRetriever:
    def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)
        self.doc_embeddings = None
        self.doc_ids = None

    def fit(self, doc_ids, docs):
        self.doc_ids = doc_ids
        self.doc_embeddings = self.model.encode(
            docs, convert_to_numpy=True, normalize_embeddings=True
        )

    def search(self, query: str, top_k=700):
        q_emb = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )
        scores = np.dot(q_emb, self.doc_embeddings.T)[0]
        idx = np.argsort(scores)[::-1][:top_k]
        return [(self.doc_ids[i], scores[i]) for i in idx]

lang = "jp"
query_ids, query_texts = load_queries("data/coliee-2023/data-preprocess/combined_en_jp_train_data.json", lang)
doc_ids, docs = load_articles("data/coliee-2023/data-preprocess/stage-2-corpus.json", lang)

biencoder = BiEncoderRetriever()
biencoder.fit(doc_ids, docs)
biencoder_results = [biencoder.search(q) for q in query_texts]
save_results("outputs/runs/biencoder/coliee-23/jp_23_train_results.json", query_ids, query_texts, biencoder_results, "BIENCODER")
#sentence-transformers/bert-base-multilingual-cased
#sentence-transformers/paraphrase-multilingual-mpnet-base-v2
