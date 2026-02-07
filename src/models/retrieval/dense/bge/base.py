import json
from sentence_transformers import InputExample

def build_pairs_bge(
    data_path,
    article_map,
    lang="en"
):
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pairs = []

    QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

    for item in data:
        # raw query theo ngôn ngữ
        if lang == "en":
            raw_query = item["query"]
        elif lang == "jp":
            raw_query = item["jp_query"]
        else:
            raise ValueError("lang must be en or jp")

        query = QUERY_INSTRUCTION + raw_query

        """pos_id = item["positive"]["article_id"]
        pos_text = article_map[pos_id]

        pairs.append(
            InputExample(texts=[query, pos_text])
        )"""

        for art in item["relevant_articles"]:
            pos_text = art["article_content"]

            pairs.append(
                InputExample(texts=[query, pos_text])
            )

    return pairs

def build_pairs_hadneg_bge(
    data_path,
    article_map,
    lang="en"
):
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pairs = []

    QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

    for item in data:
        # raw query theo ngôn ngữ
        if lang == "en":
            raw_query = item["query"]
        elif lang == "jp":
            raw_query = item["jp_query"]
        else:
            raise ValueError("lang must be en or jp")

        query = QUERY_INSTRUCTION + raw_query

        pos_id = item["positive"]["article_id"]
        pos_text = article_map[pos_id]

        pairs.append(
            InputExample(texts=[query, pos_text])
        )

        for neg in item["hard_negatives"][:2]:
            neg_text = article_map[neg["article_id"]]

            pairs.append(InputExample(texts=[query, neg_text]))


    return pairs
