from sklearn.metrics.pairwise import cosine_similarity
from ranx import Qrels, Run, evaluate
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_similarity_scores(model, df):
    '''
    Compute similarity between user query and product_doc.
    product_doc is the addition between title and description'''
    df['product_doc'] = df['product_title'] + ' ' + df['product_description']
    query_embeddings = model.encode(df['query'].tolist(), convert_to_tensor=True)
    doc_embeddings = model.encode(df['product_title'].tolist(), convert_to_tensor=True)
    similarities_mx = cosine_similarity(np.array(query_embeddings), np.array(doc_embeddings))
    similarities_diag = np.diag(similarities_mx)
    return similarities_diag

def compute_metrics(df, similarities_diag):
    '''
    Compute ndcg, recall and mrr and return'''
    qrels_dict = {}
    run_dict = {}

    for query, group in df.groupby("query"):
        query = str(query)
        # get actuals
        qrels_dict[query] = {str(example): int(relevance) for example, relevance in zip(group["example_id"], group["relevance"])}
        
        # get scores paired to each example
        examples = group["example_id"].tolist()
        example_score_pairs = list(zip(examples, similarities_diag[:len(examples)]))

        # get predicted
        run_dict[query] = {str(example): score for example, score in example_score_pairs}

    qrels = Qrels(qrels_dict)
    run = Run(run_dict)

    ndcg10, recall10, mrr10 = evaluate(qrels, run, metrics=["ndcg@10", "recall@10", "mrr@10"])

    return ndcg10, recall10, mrr10
