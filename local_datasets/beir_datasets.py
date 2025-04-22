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

        # return {**queries_as_corpus, **self.corpus}
        return self.corpus

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

class SciFact(BEIR):
    def __init__(self):
        super().__init__("scifact")
        self.url = f"{BEIR_DATASETS_BASE_URL}/scifact.zip"
        self.out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "beir")
        self.data_path = os.path.join(self.out_dir, "scifact")
        self.load()
        self.exclude_invalid_items()

    def exclude_invalid_items(self):
        self.queries.pop("768")

class SciDocs(BEIR):
    def __init__(self):
        super().__init__("scidocs")
        self.url = f"{BEIR_DATASETS_BASE_URL}/scidocs.zip"
        self.out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "beir")
        self.data_path = os.path.join(self.out_dir, "scidocs")
        self.load()
        self.exclude_invalid_items()

    def exclude_invalid_items(self):
        self.queries.pop("a29afef550bf4edbf3293a50ef3fdb785ff1e5a3")

class NFCorpus(BEIR):
    def __init__(self):
        super().__init__("nfcorpus")
        self.url = f"{BEIR_DATASETS_BASE_URL}/nfcorpus.zip"
        self.out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "beir")
        self.data_path = os.path.join(self.out_dir, "nfcorpus")
        self.load()
        self.exclude_invalid_items()

    def exclude_invalid_items(self):
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

class FIQA(BEIR):
    def __init__(self):
        super().__init__("fiqa")
        self.url = f"{BEIR_DATASETS_BASE_URL}/fiqa.zip"
        self.out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "beir")
        self.data_path = os.path.join(self.out_dir, "fiqa")
        self.load()
        self.exclude_invalid_items()

    def exclude_invalid_items(self):
        invalid = ['101883', '486802', '38221', '133415', '205961', '42442', '430152', '584229', '238766', '585289',
                   '87369', '301479', '243478', '242979', '89669', '7915', '12229','14135', '33445', '40982', '54960', '66248', '115636',
                   '117276', '126502', '138418', '153104', '167128', '169220', '189754', '206368','207473', '215635', '237392', '248226',
                   '254541', '290110', '319894', '325407', '356535', '358213', '360125', '399083', '414219', '441462','447488', '470232',
                   '486245', '528558', '552533', '572451', '587667', '597929'
                   ]

        for i in invalid:
            self.queries.pop(i, None)
            self.corpus.pop(i, None)

class ArguAna(BEIR):
    def __init__(self):
        super().__init__("arguana")
        self.url = f"{BEIR_DATASETS_BASE_URL}/arguana.zip"
        self.out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "beir")
        self.data_path = os.path.join(self.out_dir, "arguana")
        self.load()
        self.exclude_invalid_items()

    def exclude_invalid_items(self):
        invalid = ['training-society-iasihbmubf-pro06a']

        for i in invalid:
            self.queries.pop(i, None)
            self.corpus.pop(i, None)

class TRECCOVID(BEIR):
    def __init__(self):
        super().__init__("trec-covid")
        self.url = f"{BEIR_DATASETS_BASE_URL}/trec-covid.zip"
        self.out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "beir")
        self.data_path = os.path.join(self.out_dir, "trec-covid")
        self.load()
        self.exclude_invalid_items()

    def exclude_invalid_items(self):
        invalid = ['vy9u5acm','k5a1oyee','r1644q1s','88eo0ktq','tpz4wyow','mvoq9vln','nqztn30k','j4eluall','vcb32uug','2a7kn146','y4lfc4va',
            '2qf7kdtb','fzh27v3x','098fpoec','hay19wnl','1m4bmia8','cgcvfftf','tlvgys96','fz9p6gyn','4ewemyny','e5qacf03','0e0stvsd',
            'k3opc553','fafz60od','4zbh8o4s','m4fhtvyi','ulu28g3r','iuw0wlno','nkgs2iv6','vboa8xn9','xvzmhbm7','2307yevz','1x5uthma',
            'xqn1gm50','035fpn54','8vbygrss','q4c1vlt9','puefswhz','aqjcxjfq','aksv9sqg','2goaji9g','xzobolak','xat7b13b','jppi8rrh',
            'cqfgjzi2','qf6x8no3','0qf7xkwh','hc31mov3','rtgl9ft0','pyhx7oqs','oc7b46j5','8ba8pcho','7l91ktd8','wr78uelu','sblnh4c3',
            'ackvbi1g','48xf7piy','6q4yhekq','fegzawne','h8meqofa','46gkl2xt','9ujmslv0','rw40jccq','c6evx37l','hqs94hfa','i8cvlrdz',
            'l5kmaa5n','weie8p6p','9iuu8tb9','o6jwdrhg','d3pqa7w1','5a8tfjeo','nv3s24s6','x5ffqo8t','3xk73fiu','l3hm1s15','qmk11h0c',
            'e0rscasj','rtg7fs82','9uvkok8h','qi6mz6j6','hald4eeu','9ukatfop','r50kg365','1sj05rmc','pfig5o3t','opapneti','u9vy1b0l',
            'nmna0cva','eajcv0n1','q1pxr9m7','wf0ax8ky','ddwz8lc8','59eumyiv','02qd3ifl','f6ac3x6j','3sfg1nqk','vte4wnjf','zm6au11f',
            'mcyfv6x5','o3u10yud','2piml3jl','aei5jyp8','rsd49oeh','z3n0l9ay','5lua646m','suosfunh','bngoaror','pxx74evg','tm61vhpt',
            'kmsnqej0','22i9ci72','9iz23n0z','9buy4mor','x1p212w0','5okgevo2','hyifcqvx','geq0uecd','ffsch4n1','544cbc4z','mbachzoz',
            'hmsry13z','knal5kiv','01eufl5r','y2tcqh16','ftil1ryd','titoztzh','ldc5659f','n8vwqjf2','0yzkqykg','exy6cv30','13fr5jwb',
            '3334toh4','rr4yep41','ldvwjk76','rri8yeud','6xdhqjpz','y26w5rgs','yfrx789e','eiyqm5oh','p8zp4ose','qdxhyqjz','fq4ix5o6',
            'wzdb55nf','jbitv26a','279be6hp','cr035n0w','efeg1xhi','8vjsqp2d','ddmq7eix','ni68x09h','1dxbwn8r','qwm5dpu1','s6v298ns',
            'm9jlq1hv','yozipvpp','dpj4tc5a','b0dgkx6d','1d597ggf','9x7sfl7i','p67wyg5r','yx0busge','f3yln902','t3x1254w','3lkvxedj',
            'qbcnd0so','2aabks2n','djqtctet','8dcztawv','aynft6kd','22keyj91','r9y6ghrz','7ttb8wkd','3u7l856t','5kplslxw','yxkub8id',
            'b2ah7a82','suqkyotv','zednsmnj','00ajdmac','8mg2u26a','dnue4c0m','wj02rg4j','oa01gfml','tzpg3y9s','ewxxobo2','xm7hq1vq',
            'odx9p330','9500pwcy','1l887fyy','sbi5wyye','mxqtn537','npifpn3s','9wnmdvg5','wcpjph7b','ucc5btld','kkkw3ye6','70kjhvyy',
            'b074w9sv','4bj099ie','gvpbqslw','qo5s4u9e','5m17xvyi','76bv22m2','4kt5wnmg','obupqcua','0nnbbhbe','qsz11wqp','14pc9urc',
            'enbwvymu','yof80wia','l52kwq4u','nzxz61yw','zco0zpax','v9krxyx3','545qix1e','2zu6fyao','kf81eiw8','zjmhuan0','26869uo0',
            'qvcns6tc','qrlaoxo2','otpjh7s2','4tvhqmvc','lkyd6ffu','shdbjkuv','28hszy1a','ilsf23w1','h484c72f','z02wmrjh','n82tuk6k',
            'ub22crpe','fsfvkj3o','abiyvgfh','neqvr487','0aft8ruk','4p42lb3e','qtuz4iz6','73jueccn','2deix9t3','ll7e0gua','gnz9cwtm',
            'g305kvxv','k4opkbmr','asskv25i','6nc8xuxl','chmwmae4','j0817b9p','8cc3ajfb','3m4b9uau','g5vlznym','4t07iqwh','i37kn591',
            'n2vz7a1q','zse2igvv','q9flp244','lmhwt2a0','96dzdrtv','fygto0kr','0w29kvwb','28wal4p1','8p1a23yv','kdol990p','4ejh50ab',
            '8fbz3ikw','iq26puxa','vsy9utsg','61utt66s','6fgwasq2','exypmedj','9ccplhbe','n09hkowc','kxqcvssi','vo3krt28','s0mulgty',
            'emiwq67p','drhae7vy','yy258hj2','y17iq0f9','hvqltef7','5c5l7e9h','hh59ffsd','nf3bezwv','yq343cd5','k4x86f42','hnzw0lhp',
            'li6b17ov','zmhq2riv','q412fnv6','oesv1h6s','yo8375aq','ii0t5blp','ouia3rix','v2tp14pu','iryaz269','kv46za5l','matvcycp',
            '3nic2z80','lstsqyt8','cewu19h9','e7tl6ekf','bt5snhu5','c6i84vad','1kc3mmoc','4jwm1fzn','1lrez8n2','wsy6f4xr','syg2hwa2',
            '00ecwiic','ubrjaxb3','aq63jl63','p1mf07q8','ctgvk8up','odwrz8tl','3s4by6o4','ivoyalxq','1rld600i','fnkh0ulv','nggt6w2b',
            '1a4xj295','fnprk5u0','ggim0trd','5eviba1t','noryz6p1','xr1abtat','wjc6tc15','1stsoa98','kbudwh9e','bt5r8erl','be7z7vgh',
            '8miur67a','clqps03l','w5lkel28','21rlyd5b','i008c10w','hfmtdq6h','arr5tmhw','ut9uxu3d','uw16lhgp','vjvdjud3','c1tamfsf',
            'l9ynmiou','wano389t','1u2krb8q','yuiiosfu','wu3a6bvd','b6h0o0q4','9k9cj64t','bxkg4uf0','rgvr689i','o3zksn3g','b55o8ysl',
            'l1rbyojd','7lyh9byx','yindercz','a206ovml','swommq4w','e66s7chx','s32d15iz','ckvx2k56','stedi1x5','h99ywt4z','n0u5xjxt',
            'b2cm88gd','4aczx8o1','ml2zoxzx','ag3bwcjg','rzzik0yz','ubw2yl06','bj9wg9di','4sq86n9e','014g9lov','h151i11b','arg07b3h',
            '90rndrso','vheqc6b7','2xlym01m','6kck6qti','grmd39i7','ybhcwln6','btuciicn','y3uqik1o']
        for i in invalid:
            self.queries.pop(i, None)
            self.corpus.pop(i, None)

class Quora(BEIR):
    def __init__(self):
        super().__init__("quora")
        self.url = f"{BEIR_DATASETS_BASE_URL}/quora.zip"
        self.out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "beir")
        self.data_path = os.path.join(self.out_dir, "quora")
        self.load()
        self.exclude_invalid_items()

    def exclude_invalid_items(self):
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

class WebisTouche2020(BEIR):
    def __init__(self):
        super().__init__("webis-touche2020")
        self.url = f"{BEIR_DATASETS_BASE_URL}/webis-touche2020.zip"
        self.out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "beir")
        self.data_path = os.path.join(self.out_dir, "webis-touche2020")
        self.load()
        self.exclude_invalid_items()

    def exclude_invalid_items(self):
        invalid = []

        for i in invalid:
            self.queries.pop(i, None)
            self.corpus.pop(i, None)