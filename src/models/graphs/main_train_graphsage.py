import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from prepare_graph import prepare_graph, build_training_pairs
from train_graphsage import train_graphsage, load_training_triples
import torch

import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    set_seed(42)
    edges_path = "data/graph/legal-graph-2023-w4-in-ex.json"
    article_path = "data/coliee-2023/data-preprocess/stage-2-corpus.json"
    label_path = "data/coliee-2023/data-spliter/en_jp_train_set.json"

    a_ids, articles, article2id, id2article, edge_index = prepare_graph(
        edges_path,
        article_path
    )

    #training_pairs = build_training_pairs(label_path)
    #print("Num training pairs:", len(training_pairs))

    hardneg_path = "data/hard-negative-bge/hardnegative-1-2-bge.json"
    training_triples = load_training_triples(hardneg_path)

    print("Num training triples:", len(training_triples))

    graphsage = train_graphsage(
      model_path="outputs/runs/bge-biencoder/en",
      articles=articles,
      training_triples=training_triples,
      article2id=article2id,
      edge_index=edge_index,
      epochs=30
    )

    torch.save(graphsage.state_dict(), "outputs/runs/graph/graphsage_model.pt")
    print("Saved GraphSAGE model")