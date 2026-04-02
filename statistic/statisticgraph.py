import json
import networkx as nx

""" 
    Count node(total, in, ex)
    avg degree
"""

with open("coliee25/data/graph/legal-graph-2025-external.json", "r", encoding="utf-8") as f:
    internal = json.load(f)

with open("coliee25/data/graph/legal-graph-2025-internal-w4.json", "r", encoding="utf-8") as f:
    external = json.load(f)

with open("coliee25/data/civil_code_parsed.json", "r", encoding="utf-8") as f:
    articles = json.load(f)

all_nodes = set()

for article in articles:
    all_nodes.add(article["article_id"])

for src, targets in internal.items():
    all_nodes.add(src)
    all_nodes.update(targets)

for src, targets in external.items():
    all_nodes.add(src)
    all_nodes.update(targets)

internal_edges = []

for src, targets in internal.items():
    for tgt in targets:
        internal_edges.append((src, tgt))

num_internal_edges = len(internal_edges)
external_edges = set()

for src, targets in external.items():
    for tgt in targets:
        edge = tuple(sorted([src, tgt]))
        external_edges.add(edge)

num_external_edges = len(external_edges)

G = nx.Graph()

G.add_nodes_from(all_nodes)

for u, v in internal_edges:
    G.add_edge(u, v)

for u, v in external_edges:
    G.add_edge(u, v)



num_nodes = G.number_of_nodes()

total_edges = G.number_of_edges()

avg_degree = sum(dict(G.degree()).values()) / num_nodes


print(f"Number of nodes: {num_nodes}")
print(f"Number of internal edges: {num_internal_edges}")
print(f"Number of external edges: {num_external_edges}")
print(f"Average node degree: {avg_degree:.4f}")