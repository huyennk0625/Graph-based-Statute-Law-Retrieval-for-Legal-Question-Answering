import json
from sentence_transformers import InputExample

def load_articles(path):
    with open(path, "r", encoding="utf-8") as f:
        articles = json.load(f)

    article_map_en = {}
    article_map_jp = {}

    for a in articles:
        aid = a["article_id"]
        article_map_en[aid] = a["article_content_en"]
        article_map_jp[aid] = a.get("article_content_jp", "")

    return article_map_en, article_map_jp

def build_triplets(
    data_path,
    article_map,
    lang="en"   # "en" or "jp"
):
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    triplets = []

    for item in data:
        # chọn query theo ngôn ngữ
        if lang == "en":
            query = item["query"]
        elif lang == "jp":
            query = item["jp_query"]
        else:
            raise ValueError("lang must be en or jp")

        pos_id = item["positive"]["article_id"]
        pos_text = article_map[pos_id]

        for neg in item["hard_negatives"]:
            neg_id = neg["article_id"]
            neg_text = article_map[neg_id]

            triplets.append(
                InputExample(texts=[query, pos_text, neg_text])
            )

    return triplets

def build_pairs(
    data_path,
    article_map,
    lang="en"   # "en" or "jp"
):

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pairs = []

    for item in data:
        # chọn query theo ngôn ngữ
        if lang == "en":
            query = item["query"]
        elif lang == "jp":
            query = item["jp_query"]
        else:
            raise ValueError("lang must be en or jp")

        pos_id = item["positive"]["article_id"]
        pos_text = article_map[pos_id]

        # chỉ tạo pair (query, positive)
        pairs.append(
            InputExample(texts=[query, pos_text], label=1.0)
        )

    return pairs
