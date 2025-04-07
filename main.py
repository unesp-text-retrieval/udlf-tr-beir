from enum import Enum
from datasets.beir_datasets import BEIR
from models.BM25 import BM25Model
import pandas as pd

class Models(Enum):
    BM25 = "bm25"
    SBERT = "sbert"


k_values = [3, 5, 10, 25, 50, 75, 100]
datasets = [
    "scifact"
]

if __name__ == "__main__":

    # List to store dataframes
    results = []

    for name in datasets:
        corpus, queries, qrels = BEIR(name).load()

        for model_name in Models:
            if model_name == Models.BM25:
                #
                # Begin model BM25 as is
                #
                model = BM25Model(
                    name,
                    corpus,
                    queries,
                    qrels,
                    k_values
                )
                model.retrieve()
                model.evaluate()

                #
                # # Append the dataframe to the results list
                #
                results.append(model.dataframe)

                #
                # Begin calculating UDLF on top of previous results
                #
                model.name = "BM25+UDLF"
                model.udlf_run()
                model.evaluate_udlf()

                #
                # # Append UDLF the dataframe to the results list
                #
                results.append(model.dataframe)

    # Concatenate all dataframes into a single dataframe
    final_results = pd.concat(results, ignore_index=True)
    print(final_results)
