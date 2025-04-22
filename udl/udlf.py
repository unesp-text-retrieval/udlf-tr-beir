from pyUDLF import run_calls as udlf
from beir.retrieval.evaluation import EvaluateRetrieval
from udl.input_type import InputType
from udl.helpers.config import UDLFConfigHelper
from abc import ABC, abstractmethod
from models.commons.output import CommonOutput

class UDLFConfig(ABC):
    @abstractmethod
    def params(self):
        pass

class UDLF(CommonOutput):
    binary_path = None

    @staticmethod
    def set_binary_path(path: str):
        """
        Configure the path to the UDLF binary.
        """
        UDLF.binary_path = path

    def __init__(self, beir_local_datasets_path: str, dataset_name: str):
        self.beir_local_datasets_path = beir_local_datasets_path
        self.dataset_name = dataset_name
        self.ranked_list_path = f"{self.beir_local_datasets_path}{self.dataset_name}/ranked_list.txt"
        self.lists_path = f"{self.beir_local_datasets_path}{self.dataset_name}/lists.txt"
        self.output_path = f"{self.beir_local_datasets_path}{self.dataset_name}/udlf"
        self.config_ini_path = f"{self.beir_local_datasets_path}{self.dataset_name}/config.ini"
        self.udlf_config = None

    def udlf_run(self, config: UDLFConfig):
        #
        # Binary Path and Config File Path
        #
        udlf.setBinaryPath(UDLF.binary_path)
        udlf.setConfigPath(self.config_ini_path)

        #
        # Set Parameters
        #
        self.write_udlf_ranked_list_file()
        size = self.write_udlf_lists_file()

        # Initiate config by using InputType class
        self.udlf_config = InputType(self.config_ini_path)

        params = UDLFConfigHelper.create_config(
            size=size,
            ranked_list_path=self.ranked_list_path,
            lists_path=self.lists_path,
            output_path=self.output_path
        ) | config.params()

        for key, value in params.items():
            print("Setting parameter", key, "to", value)
            self.udlf_config.set_param(key, value)

        self.response = udlf.run(
            self.udlf_config,
            get_output = True
        )

        self.output_results_path = self.response.rk_path
        self.read_results()
        
    def read_results(self):
        output_results = {}
        data = open(self.output_results_path).read()
        for line in data.splitlines():
            items = line.split(" ")
            query_id = items[0]
            doc_ids = items[1:]

            # showing failures
            if query_id not in self.results.keys():
                print(f"Query {query_id} not found in the results")
                continue

            # construct the results as BEIR expects them {query_id: {doc_id: score, ...}}
            # score will be the position
            l = len(doc_ids)
            doc_id_with_positions = {doc_id: l - i for i, doc_id in enumerate(doc_ids)}
            output_results.update({query_id: doc_id_with_positions})
        
        self.udlf_results = output_results

    def write_udlf_ranked_list_file(self) -> int:
        """
        Returns integer as dataset size.
        """
        with open(self.ranked_list_path, "w") as f:
            # log insert
            print(f"Writing ranked list file to {self.ranked_list_path}")
            sorted_query_ids = sorted(self.results.keys(), key=lambda x: str(x))
            for i, query_id in enumerate(sorted_query_ids):
                retrieved_documents = self.results[query_id]

                docs = [doc_id for doc_id, score in sorted(retrieved_documents.items(), key=lambda x: x[1], reverse=True)]

                # if query_id is not docs[0] then add it into the beginning
                first_doc = docs[0]
                if query_id != first_doc:
                    docs = [query_id] + docs

                new_line = " ".join(docs)
                if i < len(sorted_query_ids) - 1:
                    f.write(new_line + "\n")
                else:
                    f.write(new_line)

            f.close()
                    
    def write_udlf_lists_file(self):
        print(f"Writing lists file to {self.lists_path}")
        with open(self.lists_path, "w") as f:
            queries = self.results.keys()
            f.write("\n".join(sorted(queries, key=lambda x: str(x))))
            f.close()
            
        return sum(1 for _ in open(self.lists_path))
            
    def evaluate_udlf(self):
        self.evaluate(self.udlf_results)

    def retrieve(self):
        self.results = self.retriever.retrieve(
            queries=self.dataset.queries_of_queries_and_docs(),
            corpus=self.dataset.corpus_of_corpus_and_queries()
        )

    def evaluate(self, external_results = None):
        self.ndcg, self._map, self.recall, self.precision = self.retriever.evaluate(
            self.dataset.qrels,
            self.results if external_results is None else external_results,
            self.retriever.k_values,
            ignore_identical_ids=False
        )
