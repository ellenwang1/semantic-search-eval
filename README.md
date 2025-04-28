# semantic-search-eval
Evaluating a pre-trained model for semantic search (retrieval)

### Instructions to run code
1. git clone https://github.com/ellenwang1/semantic-search-eval.git
2. Download the two parquet files from https://github.com/amazon-science/esci-data/tree/main/shopping_queries_dataset and upload them under the src/data/ folder locally
3. Install poetry if you don't have it already: `curl -sSL https://install.python-poetry.org | python3 -`
4. Run `poetry install`
5. Run `poetry shell` to activate your environment
6. Run `python main.py` to run the pipeline and get a final output of model results in the src/reports/ folder