# Lấy code cũ từ coliee, alqac sang
# Đã fix thêm lỗi các article bị xóa
from preprocessing.enums_const import (
    ReferenceEdgeType,
    DataFormat,
)
import networkx as nx
import pickle
import re
import json
import numpy as np
import torch

num_dict_new = {"two": 2, "three": 3}
def coliee_build_legal_graph(
    statute_corpus, content_key, internal_refer_window_size, num_dict=None
):
    # preceding_regex_pattern = r"(preceding Article)"
    preceding_regex_pattern = r"(preceding\s.*Article)"
    article_regex_pattern = r"(Article [\d\-]+)"
    articles_regex_pattern = r"(Articles [\d\-]+ \w+ [\d\-]+)"
    legal_graph = nx.MultiDiGraph()

    print("Length of statute corpus: ", len(statute_corpus))
    article_id_pool = [article.get("article_id") for article in statute_corpus]

    legal_graph.add_nodes_from(article_id_pool)

    for i, article in enumerate(statute_corpus):
        text = article.get(content_key)
        article_id = article.get("article_id")
        recent_chapter_id = article.get("chapter_id")
        # Match các kiểu tham chiếu tới điều luật ngay đằng trước
        preceding_list = text.split("preceding")
        if len(preceding_list) == 1:
            pass
        else:
            for preceding in preceding_list:
                preceding = "preceding" + preceding
                match_preceding_pattern_object = re.search(
                    preceding_regex_pattern, preceding
                )
                if match_preceding_pattern_object is not None:
                    match_preceding_article_pattern_object = (
                        match_preceding_pattern_object.group()
                    )
                    match_preceding_article_pattern_object = (
                        match_preceding_article_pattern_object.split("Article")
                    )
                    match_preceding_article_pattern_object = (
                        match_preceding_article_pattern_object[0] + "Article"
                    )
                    space_list = match_preceding_article_pattern_object.split(" ")
                    if len(space_list) > 3:
                        pass
                    elif len(space_list) == 2:
                        preceding_article = statute_corpus[i - 1].get("article_id")
                        legal_graph.add_edge(
                            article_id,
                            preceding_article,
                            weight=ReferenceEdgeType.external,
                        )
                    elif len(space_list) == 3:
                        try:
                            preceding_len = num_dict_new[space_list[1]]
                            preceding_len = int(preceding_len)
                            print(article_id, preceding_len)
                            for j in range(1, preceding_len + 1):
                                preceding_article = statute_corpus[i - j].get(
                                    "article_id"
                                )
                                legal_graph.add_edge(
                                    article_id,
                                    preceding_article,
                                    weight=ReferenceEdgeType.external,
                                )
                        except:
                            pass

        # Match các kiểu tham chiếu trực tiếp tên điều luật (Article)
        match_article_pattern_object = re.findall(article_regex_pattern, text)
        for match_group in match_article_pattern_object:
            refer_article_id = match_group.strip()
            if refer_article_id in article_id_pool:
                legal_graph.add_edge(
                    article_id, refer_article_id, weight=ReferenceEdgeType.external
                )
            else:
                print("Error refer article: ", refer_article_id)

        # Match các kiểu tham chiếu trực tiếp tên điều luật (Articles)
        match_articles_pattern_object = re.findall(articles_regex_pattern, text)
        for match_group in match_articles_pattern_object:
            span_of_refer_articles = match_group.strip()

            # Trong trường hợp Articles 646 and 650 (refer đến 2 điều luật riêng biệt)
            if "and" in span_of_refer_articles:
                for id in re.findall(r"[\d\-]+", span_of_refer_articles):
                    refer_article_id = f"Article {id}"
                    if refer_article_id in article_id_pool:
                        legal_graph.add_edge(
                            article_id,
                            refer_article_id,
                            weight=ReferenceEdgeType.external,
                        )
                    else:
                        print("Error refer article: ", refer_article_id)

            # Trong trường hợp Articles 646 through 650 (refer đến một khoảng điều luật)
            elif "through" in span_of_refer_articles:
                list_id = [id for id in re.findall(r"[\d\-]+", span_of_refer_articles)]
                assert len(list_id) == 2, "Length of through span need to equal 2"
                start_article = f"Article {list_id[0]}"
                end_article = f"Article {list_id[1]}"
                start_article_pos = [
                    i
                    for i, article in enumerate(statute_corpus)
                    if article.get("article_id") == start_article
                ]

                end_article_pos = [
                    i
                    for i, article in enumerate(statute_corpus)
                    if article.get("article_id") == end_article
                ]

                if len(start_article_pos) != 1 or len(end_article_pos) != 1:
                    print("Error refer span article: ", span_of_refer_articles)
                else:
                    for i in range(start_article_pos[0], end_article_pos[0] + 1):
                        refer_article_id = statute_corpus[i].get("article_id")
                        legal_graph.add_edge(
                            article_id,
                            refer_article_id,
                            weight=ReferenceEdgeType.external,
                        )

        # Match kiểu tham chiếu tự sinh (các article gần article hiện tại)
        for refer_article_pos in range(
            max(0, i - internal_refer_window_size),
            i,
        ):
            refer_article_chapter_id = statute_corpus[refer_article_pos].get(
                "chapter_id"
            )
            if refer_article_pos != i and refer_article_chapter_id == recent_chapter_id:
                refer_article_id = statute_corpus[refer_article_pos].get("article_id")
                legal_graph.add_edge(
                    article_id, refer_article_id, weight=ReferenceEdgeType.internal
                )
    return legal_graph


def alqac_add_external_node(statue_corpus, content_key, legal_graph):
    """Thêm các cạnh internal vào trong đồ thị.
    Các cạnh external là các cạnh được nhắc tới trực tiếp trong văn bản luật.

    Args:
        statue_corpus (_type_): tập điều luật
        content_key (_type_): key sẽ sử dụng để lấy nội dung của điều luật
        legal_graph (_type_): Object networkx DiGraph
    """
    dieu_khoan_regex = r"([Đđ]iều [0-9, (và)]+) của ((?:Bộ luật)|(?:Luật)|(?:Điều lệ)|(?:Quy chế)|(?:Thông tư)|(?:Nghị định)) ([^.;\n,]+)"

    for law in statue_corpus:
        content = law.get(content_key)
        for nhom_dieu_luat_refer in re.finditer(dieu_khoan_regex, content):
            span_dieu_luat, bo_luat, ten_bo_luat = nhom_dieu_luat_refer.groups()
            tap_dieu_luat_tham_chieu = [
                str(i) for i in re.findall(r"([0-9]+)", span_dieu_luat)
            ]
            if "này" == ten_bo_luat[:3]:
                main_article_id = law.get("article_id")
                bo_luat_goc, dieu_luat_goc = main_article_id.split("_")
                for dieu_luat_tham_chieu in tap_dieu_luat_tham_chieu:
                    refer_article_id = f"{bo_luat_goc}_{dieu_luat_tham_chieu}"
                    legal_graph.add_edge(
                        main_article_id,
                        refer_article_id,
                        weight=ReferenceEdgeType.external,
                    )


def alqac_add_internal_node(statute_corpus, legal_graph, window_size):
    """Thêm các cạnh internal vào trong đồ thị.
    Các cạnh internal là các cạnh được sinh ra gần điều luật hiện tại

    Args:
        statue_corpus (_type_): tập điều luật
        content_key (_type_): key sẽ sử dụng để lấy nội dung của điều luật
        legal_graph (_type_): Object networkx DiGraph
    """
    for i in range(len(statute_corpus)):
        main_article = statute_corpus[i]
        main_article_id = main_article.get("article_id")
        for refer_article_pos in range(
            max(0, i - window_size),
            i,
        ):
            if refer_article_pos != i:
                refer_article = statute_corpus[refer_article_pos]
                refer_article_id = refer_article.get("article_id")
                legal_graph.add_edge(
                    main_article_id,
                    refer_article_id,
                    weight=ReferenceEdgeType.internal,
                )


def alqac_build_legal_graph(statute_corpus, content_key, internal_reference_wsize):
    legal_graph = nx.MultiDiGraph()
    legal_graph.add_nodes_from(
        [article.get("article_id") for article in statute_corpus]
    )
    alqac_add_internal_node(statute_corpus, legal_graph, internal_reference_wsize)
    alqac_add_external_node(statute_corpus, content_key, legal_graph)
    return legal_graph


def build_legal_graph(
    statute_corpus, data_format, internal_reference_wsize, num_dict=None
):
    if data_format == DataFormat.COLIEE:
        content_key = "article_content_en"
        return coliee_build_legal_graph(
            statute_corpus, content_key, internal_reference_wsize, num_dict
        )
    elif data_format == DataFormat.ALQAC:
        content_key = "article_content_vn"
        return alqac_build_legal_graph(
            statute_corpus, content_key, internal_reference_wsize
        )
    else:
        assert False, f"Not support Dataformat {data_format}"


def integrate_ref_information(list_gen_sample, legal_graph, statute_corpus) -> None:
    def get_article_content_by_aid(article_id):
        for article in statute_corpus:
            if article.get("article_id") == article_id:
                return article.get("article_content_jp")
        return None

    for gen_sample in list_gen_sample:
        aid = gen_sample.get("aid")
        list_ref_article = []
        for from_node, to_node in legal_graph.edges(aid):
            for i_edge, edge_data in legal_graph.get_edge_data(
                from_node, to_node
            ).items():
                # weight = legal_graph.edges[from_node, to_node].get("weight")
                weight = edge_data.get("weight")
                assert weight != -1, "Weight cannot be -1"
                ref_article_content = get_article_content_by_aid(to_node)
                list_ref_article.append(
                    {
                        "article_id": to_node,
                        "ref_type": weight.value,
                        "content": ref_article_content,
                    }
                )
        gen_sample["list_ref_article"] = list_ref_article


def count_number_of_query_has_ref(list_gen_sample):
    query_pool = set()
    ex_query_pool = set()
    for gen_sample in list_gen_sample:
        if gen_sample.get("label") == 1:
            has_external_reference = False
            for ref_article in gen_sample["list_ref_article"]:
                if ref_article["ref_type"] == ReferenceEdgeType.external.value:
                    has_external_reference = True
                    break
            if has_external_reference:
                ex_query_pool.add(gen_sample.get("qid"))
            query_pool.add(gen_sample.get("qid"))
    print("Number of query that has external ref: ", len(ex_query_pool))
    print(
        "Percent of query that has external ref: ", len(ex_query_pool) / len(query_pool)
    )


def count_avg_external_ref_by_query(list_gen_sample):
    list_n_ext_ref = []
    for gen_sample in list_gen_sample:
        n_ext_ref = len(
            [
                ref_article
                for ref_article in gen_sample["list_ref_article"]
                if ref_article["ref_type"] == ReferenceEdgeType.external.value
            ]
        )
        list_n_ext_ref.append(n_ext_ref)
    return np.mean(list_n_ext_ref)


def statistic_number_ref_corpus(legal_graph):
    cnt_internal = 0
    cnt_external = 0
    n_nodes = len(list(legal_graph.nodes))
    list_article_have_ext = set()
    ext_ref_of_article = dict()

    for u, v, i in legal_graph.edges:
        w = legal_graph.get_edge_data(u, v, i)["weight"]
        if w == ReferenceEdgeType.external:
            cnt_external += 1
            list_article_have_ext.add(u)
            if u not in ext_ref_of_article.keys():
                ext_ref_of_article[u] = {v}
            else:
                ext_ref_of_article[u].add(v)
        elif w == ReferenceEdgeType.internal:
            cnt_internal += 1

    max_ext_of_article = 0
    for u, list_ext_ref_u in ext_ref_of_article.items():
        max_ext_of_article = max(max_ext_of_article, len(list_ext_ref_u))

    print("Cnt nodes: ", n_nodes)
    print("Cnt Internal: ", cnt_internal)
    print("Cnt external: ", cnt_external)
    print("Average external: ", cnt_external / n_nodes)
    print("n-article have external ref: ", len(list_article_have_ext))
    print("Max external reference of an article: ", max_ext_of_article)


def cnt_chapter_in_corpus(statute_corpus):
    chapter_pool = set()
    for article in statute_corpus:
        chapter_pool.add(article.get("chapter_id"))
    print("Number of chapter: ", len(chapter_pool))
    print(chapter_pool)


def reformat_graph_into_json_file(legal_graph, json_file_path, internal_external=False):
    reference_dict = dict()
    for u, v, i in legal_graph.edges:
        w = legal_graph.get_edge_data(u, v, i)["weight"]
        if internal_external == False:
            if w == ReferenceEdgeType.external:
                if u not in reference_dict.keys():
                    reference_dict[u] = []
                if v not in reference_dict[u]:
                     reference_dict[u].append(v)
        else:
            if w == ReferenceEdgeType.internal:
                if u not in reference_dict.keys():
                    reference_dict[u] = []
                if v not in reference_dict[u]:
                     reference_dict[u].append(v)
    with open(json_file_path, "w", encoding="utf-8") as json_file:
        json.dump(reference_dict, json_file, ensure_ascii=False, indent=2)

def legal_graph_to_matrix(legal_graph, node_order=None, weighted=False):
    """
    Convert a legal_graph (nx.MultiDiGraph) into an adjacency matrix (torch.Tensor).

    Args:
        legal_graph (nx.MultiDiGraph): Graph returned by coliee_build_legal_graph().
        node_order (list, optional): Custom order of node IDs (e.g., article_id list).
        weighted (bool): If True, use edge weights; if False, binary adjacency (0/1).

    Returns:
        adj_matrix (torch.Tensor): Adjacency matrix [N x N].
        node_list (list): List of article_ids in order.
    """
    # Lấy danh sách node theo thứ tự cố định
    if node_order is None:
        node_list = list(legal_graph.nodes())
    else:
        node_list = node_order

    node_index = {node: idx for idx, node in enumerate(node_list)}
    n = len(node_list)
    adj_matrix = np.zeros((n, n), dtype=float)

    # Duyệt qua tất cả các cạnh
    for u, v, data in legal_graph.edges(data=True):
        if u in node_index and v in node_index:
            i, j = node_index[u], node_index[v]
            if weighted and "weight" in data:
                adj_matrix[i, j] += float(data["weight"])  # nếu có trọng số thì cộng vào
            else:
                adj_matrix[i, j] = 1.0  # nếu chỉ muốn binary adjacency thì gán 1

    adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)
    return adj_matrix, node_list

if __name__ == "__main__":
    CORPUS_PATH = "coliee25/data/civil_code_parsed.json"
    LEGAL_JSON_FILE_PATH = (
        "coliee25/data/graph/legal-graph-2025-internal-w3.json"
    )
    NUM_DICT = "data/coliee-2023/data-preprocess/uid_dictionary.json"
    with open(NUM_DICT, "r", encoding="utf-8") as f:
        num_dict = json.load(f)
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        statute_corpus = json.load(f)
        # window size = 4
        legal_graph = build_legal_graph(statute_corpus, DataFormat.COLIEE, 3, num_dict)
        # window size = 3
        #legal_graph = build_legal_graph(statute_corpus, DataFormat.COLIEE, 3, num_dict)
        statistic_number_ref_corpus(legal_graph)
        cnt_chapter_in_corpus(statute_corpus)
        # external
        #reformat_graph_into_json_file(legal_graph, LEGAL_JSON_FILE_PATH, internal_external=False)
        # internal
        reformat_graph_into_json_file(legal_graph, LEGAL_JSON_FILE_PATH, internal_external=True)

    """with open("data/coliee-2023/data-spliter/output_files/coliee-2023/legal_graph.gpickle", "wb") as f:
        pickle.dump(legal_graph, f)

    print(legal_graph)
    for u, v, data in legal_graph.edges(data=True):
        if isinstance(data.get('weight'), ReferenceEdgeType):
            data['weight'] = int(data['weight'])

    nx.write_gexf(legal_graph, "data/coliee-2023/data-spliter/output_files/coliee-2023/legal_graph.gexf")
    # In danh sách node và edge
    #print("\nNodes:", legal_graph.nodes())
    #print("Edges:", legal_graph.edges())

    adj_matrix, article_ids = legal_graph_to_matrix(legal_graph, weighted=True)
    torch.save(adj_matrix, "data/coliee-2023/data-spliter/output_files/coliee-2023/legal-graph-matrix/legal_graph_w1_matrix.pt")

    with open("data/coliee-2023/data-spliter/output_files/coliee-2023/legal-graph-matrix/legal_graph_w1_matrix_case_sequence.json", "w") as f:
        json.dump(article_ids, f)"""
