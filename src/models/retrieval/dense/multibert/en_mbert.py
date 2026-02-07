from sentence_transformers import SentenceTransformer, losses
from torch.utils.data import DataLoader
from retrieval.dense.multibert.base import load_articles, build_triplets, build_pairs

def main():
    # Load articles
    article_map_en, article_map_jp = load_articles(
        "data/coliee-2023/data-preprocess/stage-2-corpus.json"
    )

    # EN triplets
    # result lower
    """train_examples_en = build_triplets(
        data_path="outputs/hard_negative/en_23_train_bm25_pos_hardneg.json",
        article_map=article_map_en,
        lang="en"
    )

    print("EN triplets:", len(train_examples_en))"""

    train_examples_en = build_pairs(
    data_path="outputs/hard_negative_dev/en_23_train_bm25_pos_hardneg.json",
    article_map=article_map_en,
    lang="en"
    )

    # Load mBERT
    model_en = SentenceTransformer(
        "bert-base-multilingual-cased",
        #pooling_mode_mean_tokens=True
    )

    # DataLoader
    train_loader_en = DataLoader(
        train_examples_en,
        shuffle=True,
        batch_size=32
    )

    # Triplet loss
    """train_loss = losses.TripletLoss(
        model=model_en,
        distance_metric=losses.TripletDistanceMetric.COSINE
        #margin=0.3
    )"""

    train_loss = losses.CosineSimilarityLoss(model_en)

    # Train
    """model_en.fit(
        train_objectives=[(train_loader_en, train_loss)],
        epochs=5,
        warmup_steps=300,
        optimizer_params={"lr": 2e-5},
        output_path="outputs/runs/mbert-biencoder/en"
    )"""

    model_en.fit(
    train_objectives=[(train_loader_en, train_loss)],
    epochs=2,
    warmup_steps=200,
    optimizer_params={"lr": 1e-5},
    output_path="outputs/runs/mbert-biencoder/en-dev-cosine",
    checkpoint_save_steps=0,
    show_progress_bar=True
    )



if __name__ == "__main__":
    main()