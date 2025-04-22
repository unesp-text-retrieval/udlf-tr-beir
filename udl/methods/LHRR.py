from udl.udlf import UDLFConfig

class LHRRMethod(UDLFConfig):
    def __init__(self, l: int, k: int=40, t: int=2):
        """
        Constructor for the LHRRMethod class.

        Parameters:
            l (int): Size of ranked lists (must be lesser than SIZE_DATASET, this param is mandatory).
            k (int): Number of nearest neighbors. Default is 3.
            t (int): Number of iterations. Default is 2.
        """
        self.method = "LHRR"
        self.l = l
        self.k = k
        self.t = t

    def params(self):
        """
        Returns the parameters of the LHRR method as a dictionary.
        """
        return {
            "UDL_METHOD": self.method,
            "PARAM_LHRR_L": self.l,
            "PARAM_LHRR_K": self.k,
            "PARAM_LHRR_T": self.t
        }