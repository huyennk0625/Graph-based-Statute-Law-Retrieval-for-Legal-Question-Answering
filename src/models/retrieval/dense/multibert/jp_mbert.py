from retrieval.dense.multibert.base import load_articles, build_triplets, build_pairs
from sentence_transformers import SentenceTransformer, losses
from torch.utils.data import DataLoader

def main():
    article_map_en, article_map_jp = load_articles("data/coliee-2023/data-preprocess/stage-2-corpus.json")
    
    train_examples_jp = build_pairs(
    data_path="outputs/hard_negative_dev/jp_23_train_bm25_pos_hardneg.json",
    article_map=article_map_jp,
    lang="jp"
    )

    # Load fresh model (không reuse EN)
    model_jp = SentenceTransformer(
        "bert-base-multilingual-cased",
        #pooling_mode_mean_tokens=True
    )

    train_loader_jp = DataLoader(
        train_examples_jp,
        shuffle=True,
        batch_size=32
    )

    """train_loss = losses.TripletLoss(
        model=model_jp,
        distance_metric=losses.TripletDistanceMetric.COSINE,
        margin=0.3
    )"""

    train_loss = losses.CosineSimilarityLoss(model_jp)

    """model_jp.fit(
        train_objectives=[(train_loader_jp, train_loss)],
        epochs=3,
        warmup_steps=100,
        output_path="outputs/runs/mbert-biencoder/jp"
    )"""

    model_jp.fit(
    train_objectives=[(train_loader_jp, train_loss)],
    epochs=2,
    warmup_steps=200,
    optimizer_params={"lr": 1e-5},
    output_path="outputs/runs/mbert-biencoder/jp-dev-cosine",
    checkpoint_save_steps=0,
    show_progress_bar=True
    )

if __name__ == "__main__":
    main()