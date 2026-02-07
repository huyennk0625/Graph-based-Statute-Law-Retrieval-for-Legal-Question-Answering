import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

from sentence_transformers import SentenceTransformer, losses
from torch.utils.data import DataLoader
from retrieval.dense.bge.base import build_pairs_bge
from retrieval.dense.bge.createhardneg import build_triplets, build_triplets_marginmse
from retrieval.dense.multibert.base import load_articles

def train_triplet(
    model_path,
    train_examples,
    output_path="bge-triplet",
    batch_size=8,
    epochs=5,
    lr=1e-5
):
    model = SentenceTransformer(model_path)

    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=batch_size
    )

    """train_loss = losses.TripletLoss(
        model=model,
        distance_metric=losses.TripletDistanceMetric.COSINE,
        triplet_margin=0.2
    )"""
    train_loss = losses.MarginMSELoss(model)

    for name, param in model.named_parameters():
      if "encoder.layer.0" in name or "encoder.layer.1" in name:
          param.requires_grad = False

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps = int(0.1 * len(train_dataloader)),
        optimizer_params={"lr": lr},
        output_path=output_path
    )

    return model

def main():

    article_map_en, article_map_jp = load_articles(
        "coliee25/data/civil_code_parsed.json"
    )

    """train_examples = build_triplets(
        bge_path="coliee25/outputs/runs/testok66.67/infer/en_25_train_bi_encoder_bge_m3.json",
        bm25_path="coliee25/outputs/runs/bm25/en_25_train_results.json",
        data_map="coliee25/data/split-data/train.json",
        article_map=article_map_en
    )"""

    model = SentenceTransformer("./coliee25/outputs/runs/testok66.67/en-epoch2")

    train_examples = build_triplets_marginmse(
        model=model,  # teacher
        bge_path="coliee25/outputs/runs/testok66.67/infer/en_25_train_bi_encoder_bge_m3.json",
        bm25_path="coliee25/outputs/runs/bm25/en_25_train_results.json",
        data_map="coliee25/data/split-data/train.json",
        article_map=article_map_en
    )

    print(train_examples[0].texts)

    random.shuffle(train_examples)

    model = train_triplet(
        model_path="./coliee25/outputs/runs/testok66.67/en-epoch2",
        train_examples=train_examples,
        output_path="coliee25/outputs/runs/bge-triplet-final/model"
    )

if __name__ == "__main__":
    main()