import logging
from beir import LoggingHandler
from models.commons.output import CommonOutput
from udl.udlf import UDLF

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)

from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search as BM25


class BM25Model(CommonOutput, UDLF):
    def __init__(self, dataset_name: str, corpus, queries, qrels, k_values: list, initialize: bool = True):
        self.dataset_name = dataset_name
        self.corpus = corpus
        self.queries = queries
        self.qrels = qrels
        self.name = "BM25"
        self.k_values = k_values
        self.ndcg, self._map, self.recall, self.precision = None, None, None, None
        self.model = BM25(
            index_name=self.dataset_name,
            hostname="localhost",
            initialize=initialize,
            number_of_shards=4,
        )
        self.retriever = EvaluateRetrieval(self.model, k_values=self.k_values)
        self.results = None
        self.path = f"/Users/luis.venezian/Documents/Mestrado/Github/udlf-tr-beir/datasets/beir/{self.dataset_name.lower()}/"

        if self.dataset_name == "scifact":
            self.queries.pop("768")
        if self.dataset_name == "scidocs":
            self.queries.pop("a29afef550bf4edbf3293a50ef3fdb785ff1e5a3")
        if self.dataset_name == "nfcorpus":
            invalid = ['PLAIN-3382',
                       'PLAIN-741',
                       'PLAIN-2230',
                       'PLAIN-2240',
                       'PLAIN-671',
                       'PLAIN-1794',
                       'PLAIN-1940',
                       'PLAIN-1275',
                       'PLAIN-2071',
                       'PLAIN-1568',
                       'PLAIN-997',
                       'PLAIN-2124',
                       'PLAIN-771',
                       'PLAIN-3372',
                       'PLAIN-571',
                       'PLAIN-838',
                       'PLAIN-2321',
                       'PLAIN-1506',
                       'PLAIN-2281',
                       'PLAIN-924',
                       'PLAIN-1516',
                       'PLAIN-1557',
                       'PLAIN-2197',
                       'PLAIN-2880',
                       'PLAIN-2890',
                       'PLAIN-645',
                       'PLAIN-3241',
                       'PLAIN-1962',
                       'PLAIN-1353',
                       'PLAIN-946',
                       'PLAIN-1621',
                       'PLAIN-2092',
                       'PLAIN-1309',
                       'PLAIN-2209',
                       'PLAIN-1387',
                       'PLAIN-2220',
                       'PLAIN-634',
                       'PLAIN-478',
                       'PLAIN-1909',
                       'PLAIN-2408',
                       'PLAIN-583',
                       'PLAIN-1837',
                       'PLAIN-2030',
                       'PLAIN-510',
                       'PLAIN-2040',
                       'PLAIN-2009',
                       'PLAIN-603',
                       'PLAIN-1867',
                       'PLAIN-1331',
                       'PLAIN-872',
                       'PLAIN-1772',
                       'PLAIN-1183',
                       'PLAIN-660',
                       'PLAIN-691',
                       'PLAIN-817',
                       'PLAIN-1441',
                       'PLAIN-2250',
                       'PLAIN-1288',
                       'PLAIN-987',
                       'PLAIN-1752',
                       'PLAIN-1983',
                       'PLAIN-1950',
                       'PLAIN-2019',
                       'PLAIN-520',
                       'PLAIN-2134',
                       'PLAIN-1731',
                       'PLAIN-2981',
                       'PLAIN-2156',
                       'PLAIN-1342',
                       'PLAIN-966',
                       'PLAIN-1645',
                       'PLAIN-1579',
                       'PLAIN-902',
                       'PLAIN-1611',
                       'PLAIN-2145',
                       'PLAIN-2177',
                       'PLAIN-1710',
                       'PLAIN-551',
                       'PLAIN-561',
                       'PLAIN-1161',
                       'PLAIN-1008',
                       'PLAIN-1109',
                       'PLAIN-2167',
                       'PLAIN-2386',
                       'PLAIN-1601',
                       'PLAIN-1050',
                       'PLAIN-2830',
                       'PLAIN-934',
                       'PLAIN-1098',
                       'PLAIN-2061',
                       'PLAIN-2187',
                       'PLAIN-1236',
                       'PLAIN-1496',
                       'PLAIN-1972',
                       'PLAIN-1130',
                       'PLAIN-1225',
                       'PLAIN-1214',
                       'PLAIN-1485',
                       'PLAIN-850',
                       'PLAIN-913',
                       'PLAIN-1827',
                       'PLAIN-1066',
                       'PLAIN-1847',
                       'PLAIN-3261',
                       'PLAIN-1419',
                       'PLAIN-721',
                       'PLAIN-2396',
                       'PLAIN-892',
                       'PLAIN-1897',
                       'PLAIN-1088',
                       'PLAIN-792',
                       'PLAIN-882',
                       'PLAIN-1817',
                       'PLAIN-1473',
                       'PLAIN-2311',
                       'PLAIN-2271',
                       'PLAIN-2650',
                       'PLAIN-1172',
                       'PLAIN-731',
                       'PLAIN-1679',
                       'PLAIN-593',
                       'PLAIN-1363',
                       'PLAIN-1018']

            for i in invalid:
                self.queries.pop(i, None)
                self.corpus.pop(i, None)

        if self.dataset_name == "quora":
            invalid = ['449330', '113836', '257103', '119135', '82751', '229440', '43684', '256285', '99588', '203769',
                       '260659', '178982', '75059', '485828', '415084', '334468', '78035', '481842', '127361', '97519',
                       '165815', '33644', '7489', '469122', '44699', '434176', '334581', '142847', '125092', '535806',
                       '145998', '343992', '359781', '332695', '70675', '300713', '358814', '278501', '192454', '82457',
                       '241203', '51173', '477453', '24957', '254532', '213924', '43014', '31673', '95991', '60711',
                       '189033', '408822', '365622', '445944', '535494', '85605', '443618', '479830', '420848', '22501',
                       '453852', '253190', '378979', '196040', '398282', '269923', '187517', '297349', '363577',
                       '12431', '464168', '276121', '511487', '184340', '228469', '318128', '508932', '323897',
                       '325869', '414705', '457376', '416537', '196039', '259132', '151838', '329126', '339933',
                       '323090', '116587', '104591', '147639', '487651', '445307', '29880', '299771', '413771', '18800',
                       '486966', '224327', '236720', '521483', '206899', '509167', '75548', '507177', '147169',
                       '273075', '66222', '109524', '261977', '314854', '411318', '315029', '374263', '533540', '99231',
                       '55102', '265819', '305498', '245860', '214336', '183216', '440429', '527233', '481871', '39974',
                       '320602', '397598', '459915', '384924', '517423', '302745', '455848', '116984', '475728',
                       '416454', '525519', '2024', '199228', '64147', '268866', '352077', '361069', '384593', '507395',
                       '520302', '262028', '504077', '456231', '299998', '455319', '309137', '101522', '317639',
                       '320348', '251187', '373939', '54350', '245888', '338814', '528296', '363191', '241949',
                       '199949', '15753', '384856', '391616', '403497', '6548', '242544', '185212', '158858', '88682',
                       '309237', '181043', '58676', '480179', '247533', '107999', '257635', '290462', '373114', '13197',
                       '416544', '425807', '310807', '417119', '28660', '347197', '351277', '226916', '266011', '78439',
                       '455641', '50489', '479298', '130343', '177550', '182357', '162625', '353462', '64561', '298570',
                       '361146', '381206', '228340', '161617', '237093', '432997', '316383', '473704', '280509',
                       '106677', '309879', '158857', '462061', '130126', '444007', '77565', '378443', '134221',
                       '246995', '399932', '194424', '388664', '155328', '534965', '280508', '278339', '26057',
                       '299770', '266224', '290015', '422470', '360750', '80055', '209167', '255663', '206900',
                       '194227', '254533', '344084', '418159', '509991', '303150', '403473', '439029', '213609',
                       '51174']

            for i in invalid:
                self.queries.pop(i, None)
                self.corpus.pop(i, None)

        if self.dataset_name == "fiqa":
            invalid = ['101883', '486802', '38221', '133415', '205961', '42442', '430152', '584229', '238766', '585289',
                       '87369', '301479', '243478', '242979', '89669']

            for i in invalid:
                self.queries.pop(i, None)
                self.corpus.pop(i, None)

    def queries_of_queries_and_docs(self):
        """
        Function to get the queries from the queries and corpus dataset
        Args:
            queries: Queries dataset
            corpus: Corpus dataset
        Returns:
            queries + corpus
        """
        corpus_as_queries = {}

        for id, value in self.corpus.items():
            corpus_as_queries[id] = value.get("title") + " " + value.get("text")

        return {**corpus_as_queries, **self.queries}

    def corpus_of_corpus_and_queries(self):
        """
        Function to get the corpus from the corpus and queries dataset
        Args:
            corpus: Corpus dataset
            queries: Queries dataset
        Returns:
            corpus + queries
        """
        queries_as_corpus = {}

        for key, value in self.queries.items():
            queries_as_corpus[key] = {'text': value, 'title': value}

        return {**queries_as_corpus, **self.corpus}

    def pop_querie_or_corpus_id(self, id):
        """
        Function to remove the query or corpus id from the dataset
        Args:
            id: Query or corpus id
        """
        if id in self.queries:
            print(f"Removing query id {id} from queries")
            del self.queries[id]
        elif id in self.corpus:
            print(f"Removing corpus id {id} from corpus")
            del self.corpus[id]

    def retrieve(self):
        q = self.queries_of_queries_and_docs()
        c = self.corpus_of_corpus_and_queries()

        self.results = self.retriever.retrieve(
            queries=q,
            corpus=c
        )

    def evaluate(self):
        self.ndcg, self._map, self.recall, self.precision = self.retriever.evaluate(
            self.qrels,
            self.results,
            self.retriever.k_values,
            ignore_identical_ids=False
        )

    @property
    def data(self):
        return {
            "dataset": self.dataset_name,
            "k_values": self.k_values,
            "retriever": "BM25",
            "ndcg": self.ndcg,
            "map": self._map,
            "recall": self.recall,
            "precision": self.precision
        }