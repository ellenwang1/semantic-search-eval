# semantic-search-eval
Evaluating a pre-trained model for semantic search (retrieval)

### Instructions to run code
1. git clone https://github.com/ellenwang1/semantic-search-eval.git
2. Download the two parquet files from https://github.com/amazon-science/esci-data/tree/main/shopping_queries_dataset and upload them under a newly created `src/data/` folder locally so that the data paths are: `src/data/shopping_queries_dataset_examples.parquet` and `src/data/shopping_queries_dataset_products.parquet`
3. Install miniconda from `https://www.anaconda.com/docs/getting-started/miniconda/main` or anaconda from `https://www.anaconda.com/docs/getting-started/anaconda/install`
4. Create and activate a Conda environment: `conda create -n semantic-search python=3.12` and `conda activate semantic-search`
5. Run `pip install poetry` 
6. Run `poetry install --no-root`
7. Run `python main.py` to run the pipeline and get a final csv of model results in the `src/reports/` folder as `src/reports/metrics_summary.csv`