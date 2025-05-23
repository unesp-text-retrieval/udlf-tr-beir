import logging
from beir import LoggingHandler
from udl.udlf import UDLF
from local_datasets.beir_datasets import BEIR

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)

from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search as BM25


class BM25Model(UDLF):
    def __init__(self, dataset: BEIR, k_values: list, initialize: bool = True):
        self.dataset = dataset
        self.name = "BM25"
        self.k_values = k_values
        super().__init__(
            beir_local_datasets_path=self.dataset.out_dir + "/",
            dataset_name=self.dataset.dataset_name
        )
        self.ndcg, self._map, self.recall, self.precision = None, None, None, None
        self.model = BM25(
            index_name=self.dataset.dataset_name,
            hostname="localhost",
            initialize=initialize,
            number_of_shards=4,
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