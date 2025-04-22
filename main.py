from enum import Enum
from local_datasets.beir_datasets import BEIR, SciFact, SciDocs, NFCorpus, FIQA, ArguAna, TRECCOVID, WebisTouche2020
from models.BM25 import BM25Model
from udl.methods.CPRR import CPRRMethod
from udl.methods.LHRR import LHRRMethod
from udl.methods.RDPAC import RDPACMethod
from udl.udlf import UDLF
from dotenv import load_dotenv
import os
from models.SBERT import SBERTModel
import pandas as pd

def run_udlf_with_methods(model, methods):
    results = []
    for method_name, method_config in methods.items():
        model.udlf_run(config=method_config)
        model.evaluate_udlf()
        results.append(model.dataframe(method=method_name))
    return results

class Models(Enum):
    BM25 = "bm25"
    SBERT = "sbert"

k_values = [1,3,5,7,10,15,20,25,30,50,75,100]
datasets = [
    SciFact(),
    #ArguAna(),
    #NFCorpus(),
    #SciDocs()
]

if __name__ == "__main__":

    # Load environment variables from .env file
    load_dotenv()

    # Access the variables
    bin_path = os.getenv("UDLF_BINARY_PATH")
    if not bin_path:
        raise EnvironmentError("UDLF_BINARY_PATH is not set in the .env file.")

    # Configure UDLF binary path
    UDLF.set_binary_path(bin_path)

    # List to store dataframes
    results = []

    # UDLF methods to be used
    methods = {
        "CPRR": CPRRMethod(l=100, k=3, t=2),
        "LHRR": LHRRMethod(l=100, k=2, t=2),
        "RDPAC": RDPACMethod(l=100, k_start=1, k_end=3, k_inc=1, l_mult=2, p=0.60, pl=0.99)
    }

    for dataset in datasets:
        for model_name in Models:
            if model_name == Models.BM25:
                model = BM25Model(dataset, k_values)
                model.retrieve()
                model.evaluate()
                results.append(model.dataframe())
                results.extend(run_udlf_with_methods(model, methods))

    # Concatenate all dataframes into a single dataframe
    final_results = pd.concat(results, ignore_index=True)
    print(final_results)
    final_results.to_csv("final_results.csv", index=False)
