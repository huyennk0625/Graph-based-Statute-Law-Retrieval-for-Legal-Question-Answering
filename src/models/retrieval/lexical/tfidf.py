from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from retrieval.base import tokenize_en, tokenize_jp, load_queries, load_articles, save_results
import numpy as np

class TfidfRetriever:
    def __init__(self, lang="en"):
        self.lang = lang
        self.vectorizer = None
        self.doc_vectors = None
        self.doc_ids = None

    def fit(self, doc_ids, docs):
        self.doc_ids = doc_ids
        if self.lang == "en":
            self.vectorizer = TfidfVectorizer(tokenizer=tokenize_en)
        else:
            from sudachipy import tokenizer as tk
            from sudachipy import dictionary
            tokenizer = dictionary.Dictionary().create()
            self.vectorizer = TfidfVectorizer(
                tokenizer=lambda x: tokenize_jp(x, tokenizer)
            )
        self.doc_vectors = self.vectorizer.fit_transform(docs)

    def search(self, query: str, top_k=700):
        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.doc_vectors)[0]
        idx = np.argsort(scores)[::-1][:top_k]
        return [(self.doc_ids[i], scores[i]) for i in idx]

lang = "jp"
query_ids, query_texts = load_queries("data/coliee-2021/coliee-2021-preprocessed/combined_en_jp_train_data.json", lang)
doc_ids, docs = load_articles("data/coliee-2023/data-preprocess/stage-2-corpus.json", lang)

tfidf = TfidfRetriever(lang)
tfidf.fit(doc_ids, docs)
tfidf_results = [tfidf.search(q) for q in query_texts]
save_results("outputs/runs/tfidf/coliee-21/jp_21_train_results.json", query_ids, query_texts, tfidf_results, "TF-IDF")
