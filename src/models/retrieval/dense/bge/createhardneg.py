import json
import random
import numpy as np
from sentence_transformers import InputExample


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_rank_map(data):
    """
    Convert:
    {
      "queries": [
        {
          "query_id": ...,
          "results": [{"article_id": ...}, ...]
        }
      ]
    }
    ->
    { query_id: [article_id1, article_id2, ...] }
    """
    rank_map = {}
    for q in data["queries"]:
        qid = q["query_id"]
        rank_map[qid] = [r["article_id"] for r in q["results"]]
    return rank_map


def get_hard_negatives(candidates, positives, start=3, end=20):
    return [a for a in candidates[start:end] if a not in positives]

def get_random_negative(all_articles, positives, exclude=set()):
    candidates = list(set(all_articles) - set(positives) - set(exclude))
    return random.choice(candidates)

def build_maps_from_gold(data_path, lang="en"):
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    gt_map = {}
    query_map = {}

    QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

    for item in data:
        qid = item["id"]

        # chọn query theo ngôn ngữ
        if lang == "en":
            raw_query = item["query"]
        elif lang == "jp":
            raw_query = item["jp_query"]
        else:
            raise ValueError("lang must be en or jp")

        query_map[qid] = QUERY_INSTRUCTION + raw_query

        # lấy list positive article_id
        gt_map[qid] = [art["article_id"] for art in item["relevant_articles"]]

    return gt_map, query_map

def build_triplets(
    bge_path,
    bm25_path,
    data_map,
    article_map,
    num_dense=3,
    num_bm25=3,
    seed=42
):
    random.seed(seed)

    bge_data = load_json(bge_path)
    bm25_data = load_json(bm25_path)

    bge_rank = build_rank_map(bge_data)
    bm25_rank = build_rank_map(bm25_data)

    all_articles = list(article_map.keys())

    gt_map, query_map = build_maps_from_gold(data_map, "en")

    train_examples = []

    for qid, positives in gt_map.items():
        if qid not in bge_rank or qid not in bm25_rank:
            continue

        query_text = query_map[qid]

        dense_candidates = get_hard_negatives(
            bge_rank[qid], positives, start=8, end=40)

        bm25_candidates = get_hard_negatives(
            bm25_rank[qid], positives, start=10, end=60)
        
        dense_candidates = list(set(dense_candidates))
        bm25_candidates = list(set(bm25_candidates) - set(dense_candidates))

        if len(dense_candidates) == 0:
            dense_candidates = bm25_candidates.copy()

        if len(bm25_candidates) == 0:
            bm25_candidates = dense_candidates.copy()

        if len(dense_candidates) == 0:
            continue
        
        random.shuffle(dense_candidates)
        random.shuffle(bm25_candidates)
        
        pos_list = positives.copy()
        random.shuffle(pos_list)

        for pos in pos_list:
            pos_text = article_map[pos].strip()

            #dense_sample = dense_candidates[:num_dense]
            #bm25_sample = bm25_candidates[:num_bm25]

            dense_sample = random.sample(
                dense_candidates,
                k=min(num_dense, len(dense_candidates))
            )

            bm25_sample = random.sample(
                bm25_candidates,
                k=min(num_bm25, len(bm25_candidates))
            )

            for neg in dense_sample:
                train_examples.append(
                    InputExample(texts=[query_text, pos_text, article_map[neg]])
                )

            for neg in bm25_sample:
                train_examples.append(
                    InputExample(texts=[query_text, pos_text, article_map[neg]])
                )

            used_negs = set(dense_sample) | set(bm25_sample) | set(positives)
            #neg = get_random_negative(all_articles, positives, exclude=used_negs)

            # random
            neg1 = get_random_negative(all_articles, positives, exclude=used_negs)

            # thêm 1 easy random nữa
            neg2 = get_random_negative(all_articles, positives, exclude=used_negs | {neg1})

            train_examples.append(InputExample(texts=[query_text, pos_text, article_map[neg1]]))
            train_examples.append(InputExample(texts=[query_text, pos_text, article_map[neg2]]))
            train_examples.append(
                InputExample(texts=[query_text, pos_text, article_map[neg]])
            )

    random.shuffle(train_examples)
    print("Total triplets:", len(train_examples))

    return train_examples

def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def build_triplets_marginmse(
    model,  # teacher model
    bge_path,
    bm25_path,
    data_map,
    article_map,
    num_dense=3,
    num_bm25=3,
    seed=42
):
    random.seed(seed)

    def load_json(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def build_rank_map(data):
        rank_map = {}
        for q in data["queries"]:
            qid = q["query_id"]
            rank_map[qid] = [r["article_id"] for r in q["results"]]
        return rank_map

    def get_hard_negatives(candidates, positives, start=5, end=40):
        return [a for a in candidates[start:end] if a not in positives]

    def get_random_negative(all_articles, positives, exclude=None):
        if exclude is None:
            exclude = set()
        candidates = list(set(all_articles) - set(positives) - set(exclude))
        return random.choice(candidates)

    def build_maps_from_gold(data_path):
        data = load_json(data_path)
        gt_map, query_map = {}, {}

        QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

        for item in data:
            qid = item["id"]
            query_map[qid] = QUERY_INSTRUCTION + item["query"]
            gt_map[qid] = [art["article_id"] for art in item["relevant_articles"]]

        return gt_map, query_map

    bge_data = load_json(bge_path)
    bm25_data = load_json(bm25_path)

    bge_rank = build_rank_map(bge_data)
    bm25_rank = build_rank_map(bm25_data)

    gt_map, query_map = build_maps_from_gold(data_map)
    all_articles = list(article_map.keys())

    train_examples = []

    print("Encoding all articles...")
    article_ids = list(article_map.keys())
    article_texts = [article_map[a].strip() for a in article_ids]

    article_embs = model.encode(article_texts, convert_to_numpy=True, show_progress_bar=True)
    article_emb_map = {aid: emb for aid, emb in zip(article_ids, article_embs)}

    print("Building MarginMSE triplets...")

    for qid, positives in gt_map.items():
        if qid not in bge_rank or qid not in bm25_rank:
            continue

        query_text = query_map[qid]
        q_emb = model.encode(query_text, convert_to_numpy=True)

        dense_candidates = get_hard_negatives(bge_rank[qid], positives, 5, 40)
        bm25_candidates = get_hard_negatives(bm25_rank[qid], positives, 10, 60)

        dense_candidates = list(set(dense_candidates))
        bm25_candidates = list(set(bm25_candidates) - set(dense_candidates))

        if len(dense_candidates) == 0:
            dense_candidates = bm25_candidates.copy()
        if len(bm25_candidates) == 0:
            bm25_candidates = dense_candidates.copy()
        if len(dense_candidates) == 0:
            continue

        random.shuffle(dense_candidates)
        random.shuffle(bm25_candidates)

        pos_list = positives.copy()
        random.shuffle(pos_list)

        for pos in pos_list:
            pos_text = article_map[pos].strip()
            pos_emb = article_emb_map[pos]

            dense_sample = random.sample(
                dense_candidates,
                k=min(num_dense, len(dense_candidates))
            )

            bm25_sample = random.sample(
                bm25_candidates,
                k=min(num_bm25, len(bm25_candidates))
            )

            all_negs = dense_sample + bm25_sample

            # thêm 2 random easy negatives
            used = set(all_negs) | set(positives)
            neg1 = get_random_negative(all_articles, positives, used)
            neg2 = get_random_negative(all_articles, positives, used | {neg1})

            all_negs += [neg1, neg2]

            for neg in all_negs:
                neg_text = article_map[neg].strip()
                neg_emb = article_emb_map[neg]

                score_pos = cos_sim(q_emb, pos_emb)
                score_neg = cos_sim(q_emb, neg_emb)

                label = 5 * float(score_pos - score_neg)

                train_examples.append(
                    InputExample(
                        texts=[query_text, pos_text, neg_text],
                        label=label
                    )
                )

    random.shuffle(train_examples)
    print("Total MarginMSE samples:", len(train_examples))

    return train_examples