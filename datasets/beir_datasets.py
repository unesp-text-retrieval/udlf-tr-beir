from beir import util
import os
import pathlib
from beir.datasets.data_loader import GenericDataLoader

BEIR_DATASETS_BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"

class BEIR:
    def __init__(self, dataset_name):
        super().__init__()
        self.corpus = None
        self.queries = None
        self.qrels = None
        self.dataset_name = dataset_name

    def load(self):
        self.url = f"{BEIR_DATASETS_BASE_URL}/{self.dataset_name}.zip"
        self.out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "beir")
        self.data_path = os.path.join(self.out_dir, self.dataset_name)

        if not os.path.exists(self.data_path):
            self.data_path = util.download_and_unzip(self.url, self.out_dir)

        self.corpus, self.queries, self.qrels = GenericDataLoader(self.data_path).load(split="test")
        return self.corpus, self.queries, self.qrels
