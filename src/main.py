from preprocess import load_dataset, preprocess_dataset
from evaluation import compute_similarity_scores, compute_metrics
from sentence_transformers import SentenceTransformer
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    ''' Main function structure:
    1. Choose and load model
    2. Load dataset
    3. Preprocess dataset
    4. Evaluate model
    5. Model evaluation visualisations and examples
    '''

    # 1. choose and load model
    chosen_model = 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1'
    model = SentenceTransformer(chosen_model)

    # 2. load dataset, parameterise for split and locale
    df_train = load_dataset(split='train', product_locale='us')
    logging.info(f"Dataset loaded with {len(df_train)} rows.")

    # 3. preprocess dataset, isolate query, title, description, relevance
    # df_clean = preprocess_dataset(df)
    df_train_clean = preprocess_dataset(df_train)
    logging.info(f"Cleaned dataset rows: {len(df_train_clean)}")
    logging.info(f"Unique queries: {df_train_clean["query_id"].nunique()}")
    # df_test_clean = preprocess_dataset(df_test)

    # 4. evaluate model, get similarity scores
    similarity_scores = compute_similarity_scores(model, df_train_clean)
    logging.info("Similarity calculation complete")
    df_train_clean['similarity_scores'] = similarity_scores
    ndcg10_train, recall10_train, mrr10_train = compute_metrics(df_train_clean)
    logging.info(f"NDCG Metric train: {ndcg10_train}.")
    logging.info(f"Recall Metric train: {recall10_train}.")
    logging.info(f"MRR Metric train: {mrr10_train}.")

    # Do the same thing for the test set
    df_test = load_dataset(split='test', product_locale='us')
    df_test_clean = preprocess_dataset(df_test)
    similarity_scores = compute_similarity_scores(model, df_test_clean)
    df_test_clean['similarity_scores'] = similarity_scores
    ndcg10_test, recall10_test, mrr10_test = compute_metrics(df_test_clean)
    logging.info(f"NDCG Metric train: {ndcg10_test}.")
    logging.info(f"Recall Metric train: {recall10_test}.")
    logging.info(f"MRR Metric train: {mrr10_test}.")

    metrics_dict = {
        "ndcg10_train": ndcg10_train,
        "recall10_train": recall10_train,
        "mrr10_train": mrr10_train,
        "ndcg10_test": ndcg10_test,
        "recall10_test": recall10_test,
        "mrr10_test": mrr10_test,
    }
    metrics_df = pd.DataFrame(metrics_dict)

    metrics_df.to_csv('reports/metrics_summary.csv')


if __name__ == "__main__":
    main()
