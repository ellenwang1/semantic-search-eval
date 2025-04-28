from preprocess import load_dataset, preprocess_dataset
from evaluation import compute_similarity_scores, compute_metrics
from visualisation import visualise_model_performance
from sentence_transformers import SentenceTransformer
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
    #TODO: add decision criteria after working e2e
    #TODO: add file input to change out models?
    chosen_model = 'sentence-transformers/paraphrase-MiniLM-L6-v2'
    model = SentenceTransformer(chosen_model)
    logging.info(f"{model} loaded with info: {model.model_card_data}")

    # 2. load dataset, parameterise for split and locale
    df = load_dataset(split='all', product_locale='us')
    df_train = load_dataset(split='train', product_locale='us')
    df_test = load_dataset(split='test', product_locale='us')
    logging.info(f"Dataset loaded with {len(df)} rows.")

    # 3. preprocess dataset, isolate query, title, description, relevance
    df_clean = preprocess_dataset(df)
    df_train_clean = preprocess_dataset(df_train)
    df_test_clean = preprocess_dataset(df_test)

    # 4. evaluate model, get similarity scores
    similarity_scores = compute_similarity_scores(df_train_clean)
    ndcg10, recall10, mrr10 = compute_metrics(df_train_clean, similarity_scores)

    #TODO: evaluate on test?

    #5. create visualisations for final metrics
    visualise_model_performance()

if __name__ == "__main__":
    main()