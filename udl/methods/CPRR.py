from udl.udlf import UDLFConfig

class CPRRMethod(UDLFConfig):
    def __init__(self, l: int, k: int=3, t: int=2):
        """
        Constructor for the CPRRMethod class.

        Parameters:
            l (int): Size of ranked lists (must be lesser than SIZE_DATASET, this param is mandatory).
            k (int): Number of nearest neighbors. Default is 3.
            t (int): Number of iterations. Default is 2.
        """
        self.method = "CPRR"
        self.l = l
        self.k = k
        self.t = t

    def params(self):
        """
        Returns the parameters of the CPRR method as a dictionary.
        """
        return {
            "UDL_METHOD": self.method,
            "PARAM_CPRR_L": self.l,
            "PARAM_CPRR_K": self.k,
            "PARAM_CPRR_T": self.t
        }