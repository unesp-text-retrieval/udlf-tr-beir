import logging
from beir import LoggingHandler
from models.commons.output import CommonOutput
from udl.udlf import UDLF
from local_datasets.beir_datasets import BEIR

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)

from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval import models
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES


class SBERTModel(UDLF):
    def __init__(self, dataset: BEIR, k_values: list):
        self.dataset = dataset
        self.name = "sbert"
        self.k_values = k_values
        super().__init__(
            beir_local_datasets_path=self.dataset.out_dir + "/",
            dataset_name=self.dataset.dataset_name
        )
        self.ndcg, self._map, self.recall, self.precision = None, None, None, None
        self.model = DRES(
            models.SentenceBERT("all-MiniLM-L6-v2"),
            batch_size=512
        )
        self.retriever = EvaluateRetrieval(self.model, k_values=self.k_values)
        self.results = None

    @property
    def data(self):
        return {
            "dataset": self.dataset.dataset_name,
            "k_values": self.k_values,
            "retriever": "BM25",
            "ndcg": self.ndcg,
            "map": self._map,
            "recall": self.recall,
            "precision": self.precision
        }