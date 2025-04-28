from utils import map_esci
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__) 

def load_dataset(split, product_locale):
    '''
    Output subsection of data depending on what
    split and locale is defined
    
    split: ['all', 'train', 'test']
    locale: ['us', 'all']
    '''
    df_examples = pd.read_parquet('./data/shopping_queries_dataset_examples.parquet')
    df_products = pd.read_parquet('./data/shopping_queries_dataset_products.parquet')
    df_sources = pd.read_csv("./data/shopping_queries_dataset_sources.csv")
    
    # https://github.com/amazon-science/esci-data: suggested filter for task 1: Query-Product Ranking 
    # Query-Product Ranking: Given a user specified query and a list of matched products, the goal of this 
    # task is to rank the products so that the relevant products are ranked above the non-relevant ones.
    df_examples_products = pd.merge(
        df_examples,
        df_products,
        how='left',
        left_on=['product_locale','product_id'],
        right_on=['product_locale', 'product_id']
    )

    # take the small version for this task
    df_task_1_all = df_examples_products[df_examples_products["small_version"] == 1]
    df_task_1_train = df_task_1_all[df_task_1_all["split"] == "train"]
    df_task_1_test = df_task_1_all[df_task_1_all["split"] == "test"]

    # split into locale
    df_task_1_all_us = df_task_1_all[df_task_1_all['product_locale'] == 'us']
    df_task_1_train_us = df_task_1_train[df_task_1_train['product_locale'] == 'us']
    df_task_1_test_us = df_task_1_test[df_task_1_test['product_locale'] == 'us']

    if split == 'train' and product_locale == 'us':
        return df_task_1_train_us
    elif split == 'train' and product_locale =='all':
        return df_task_1_train
    elif split == 'test' and product_locale == 'us':
        return df_task_1_test_us
    elif split == 'test' and product_locale == 'all':
        return df_task_1_test
    elif split == 'all' and product_locale == 'us':
        return df_task_1_all_us
    else:
        return df_task_1_all
    
def preprocess_dataset(df):
    '''
    Assert data is clean, no outliers in training.
    Return df so that only ['query', 'title', 'description', 'relevance'] are in the columns.'''

    # Ensure no duplicates
    duplicates = df[df.duplicated(subset=["query", "product_title"])]
    logging.info(f'There are {len(duplicates)} duplicates in this dataset.')

    # TODO: Ensure no really short queries that are not very specific

    # Map ESCI to relevance
    esci_weighting = {
        'E': 3,
        'S': 2,
        'C': 1,
        'I': 0
    }
    df['relevance'] = df['esci_label'].map(esci_weighting)
    # Isolate query, title, description, relevance
    preprocessed_df = df[['query', 'product_title', 'product_description', 'relevance']]
    
    logging.info('We are expecting four columns in the preprocessed df.')
    logging.info(f'There are {len(preprocessed_df)} columns in the preprocessed df')

    return preprocessed_df
