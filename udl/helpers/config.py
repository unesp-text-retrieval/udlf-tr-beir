class UDLFConfigHelper:
    @staticmethod
    def create_config(
            size: int,
            ranked_list_path: str,
            lists_path: str,
            output_path: str
    ) -> dict:
        return {
            "UDL_TASK": "UDL",
            "SIZE_DATASET": size,
            "INPUT_FILE": ranked_list_path,
            "INPUT_FILE_LIST": lists_path,
            "OUTPUT_FILE_PATH": output_path
        }