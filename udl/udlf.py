from pyUDLF import run_calls as udlf
from beir.retrieval.evaluation import EvaluateRetrieval
from udl.input_type import InputType
import time

class UDLF:
        
    def __init__(self):
        self.udlf_results = None
        self.output_results_path = None
        self.udlf_response = None
 
    def udlf_run(self):
        datasets_path = "/Users/luis.venezian/Documents/Mestrado/Github/udlf-tr-beir/datasets/beir/"
        udlf.setBinaryPath("/Users/luis.venezian/Documents/Mestrado/Github/pyUDLF/udlf")
        udlf.setConfigPath(datasets_path + "config.ini")
        self.write_udlf_ranked_list_file()
        size = self.write_udlf_lists_file()
        
        udlf_config = InputType()
        udlf_config.set_param("UDL_METHOD", "CPRR")
        udlf_config.set_param("SIZE_DATASET", size)
        udlf_config.set_param("UDL_TASK", "UDL")
        udlf_config.set_param("INPUT_FILE", datasets_path + f"{self.dataset_name}/ranked_list.txt")
        udlf_config.set_param("INPUT_FILE_LIST", datasets_path + f"{self.dataset_name}/lists.txt")
        udlf_config.set_param("OUTPUT_FILE_PATH", datasets_path + f"{self.dataset_name}/udlf")
        
        print("############################")
        print("INPUT_FILES ", udlf_config.get_input_files())
        print("LISTS FILES ", udlf_config.get_lists_file())
        print("OUTPUT FILE PATH", udlf_config.get_output_file_path())
        print("############################")
        
        self.udlf_response = udlf.run(
            udlf_config, 
            get_output = True
        )
        
        self.output_results_path = self.udlf_response.rk_path
        print("############################\nOUTPUT FILE PATH", self.output_results_path)
        
        if self.dataset_name == "quora":
            time.sleep(60)
            
        self.read_results()
        
    def read_results(self):
        output_results = {}
        data = open(self.output_results_path).read()
        for line in data.splitlines():
            items = line.split(" ")
            query_id = items[0]
            doc_ids = items[1:]
            
            # handling failures
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
        Writes the ranked list of documents for each query to a file.

        Parameters:
            results (dict): A dictionary where keys are query IDs and values are dictionaries of document IDs and scores.
            file_path (str): The path to the file where the ranked list will be written.
            
        Returns integer as dataset size.
        """
        
        
        with open(self.path + "ranked_list.txt", "w") as f:
            sorted_query_ids = sorted(self.results.keys(), key=lambda x: str(x))
            for i, query_id in enumerate(sorted_query_ids):
                retrieved_documents = self.results[query_id]
                # docs = [query_id] + [doc_id for doc_id, score in retrieved_documents.items()]
                docs = [doc_id for doc_id, score in retrieved_documents.items()]
                new_line = " ".join(docs)
                if i < len(sorted_query_ids) - 1:
                    f.write(new_line + "\n")
                else:
                    f.write(new_line)
                    
    def write_udlf_lists_file(self):
        """
        Writes the sorted list of query IDs to a file.

        Parameters:
            results (dict): A dictionary where keys are query IDs.
            file_path (str): The path to the file where the query IDs will be written.
        """
        with open(self.path + "lists.txt", "w") as f:
            queries = self.results.keys()
            f.write("\n".join(sorted(queries, key=lambda x: str(x))))
            
        return sum(1 for _ in open(self.path + "lists.txt"))
            
    def evaluate_udlf(self):
        """
        Evaluate the UDLF results using BEIR's evaluation metrics.
        """
        self.ndcg, self._map, self.recall, self.precision = EvaluateRetrieval.evaluate(
            self.qrels, 
            self.udlf_results,
            k_values=self.k_values,
            ignore_identical_ids=False
        )