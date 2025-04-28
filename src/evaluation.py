import torch.nn.functional as F
from ranx import Qrels, Run, evaluate
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_similarity_scores(model, df):
    '''
    Compute similarity between user query and product_doc.
    product_doc is the addition between title and description'''
    query_embeddings = model.encode(df['query'].tolist(), convert_to_tensor=True).cpu()
    doc_embeddings = model.encode(df['product_doc'].tolist(), convert_to_tensor=True).cpu()
    logging.info("Embeddings complete")
    logging.info(f'Query shape: {query_embeddings.shape}')
    logging.info(f'Doc shape: {doc_embeddings.shape}')
    similarities_diag = F.cosine_similarity(query_embeddings, doc_embeddings)
    logging.info(f'Similarities matrix shape: {similarities_diag.shape}')
    # convert back to np for ease of calculations later
    similarities_np = similarities_diag.cpu().numpy()
    similarities_np = np.round(similarities_np, 2)
    return similarities_np

def compute_metrics(df):
    '''
    Compute ndcg, recall and mrr and return'''
    qrels_dict = {}
    run_dict = {}

    for query_id, group in df.groupby("query_id"):
        query_id = str(query_id)
        # get actuals
        qrels_dict[query_id] = {str(doc_id): int(relevance) for doc_id, relevance in zip(group["example_id"], group["relevance"])}
        
        # get scores paired to each doc
        docs = group["example_id"].tolist()
        similarity_scores = group['similarity_scores'].tolist()
        doc_score_pairs = list(zip(docs, similarity_scores))

        # get predicted
        run_dict[query_id] = {str(doc): score for doc, score in doc_score_pairs}

    qrels = Qrels(qrels_dict)
    logging.info("Create qrels object")
    run = Run(run_dict)
    logging.info("Create run object")

    dict_results = evaluate(qrels, run, metrics=["ndcg@10", "recall@10", "mrr@10"])
    logging.info("Calculate metrics")

    ndcg10 = dict_results["ndcg@10"]
    recall10 = dict_results["recall@10"]
    mrr10 = dict_results["mrr@10"]

    return ndcg10, recall10, mrr10
