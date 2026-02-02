from rank_bm25 import BM25Okapi
from retrieval.base import tokenize_en, tokenize_jp, load_queries, load_articles, save_results
from sudachipy import tokenizer as tk
from sudachipy import dictionary
import numpy as np

class BM25Retriever:
    def __init__(self, lang="en"):
        self.lang = lang
        self.bm25 = None
        self.doc_ids = None
        self.tokenizer = None


    def fit(self, doc_ids, docs):
        self.doc_ids = doc_ids
        if self.lang == "en":
            corpus = [tokenize_en(d) for d in docs]
        else:
            self.tokenizer = dictionary.Dictionary().create()
            corpus = [tokenize_jp(d, self.tokenizer) for d in docs]
        self.bm25 = BM25Okapi(corpus)


    def search(self, query: str, top_k=700):
        if self.lang == "en":
            q = tokenize_en(query)
        else:
            q = tokenize_jp(query, self.tokenizer)
        scores = self.bm25.get_scores(q)
        idx = np.argsort(scores)[::-1][:top_k]
        return [(self.doc_ids[i], scores[i]) for i in idx]


lang = "en"
query_ids, query_texts = load_queries("data/coliee-2023/data-spliter/en_jp_test_set.json", lang)
doc_ids, docs = load_articles("data/coliee-2023/data-preprocess/stage-2-corpus.json", lang)

bm25 = BM25Retriever(lang)
bm25.fit(doc_ids, docs)
bm25_results = [bm25.search(q) for q in query_texts]
save_results("outputs/runs/bm25/coliee-23-dev/en_23_test_results.json", query_ids, query_texts, bm25_results, "BM25")
