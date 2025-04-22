from udl.udlf import UDLFConfig

class RDPACMethod(UDLFConfig):

    def __init__(self, l: int, k_start: int = 1, k_end: int = 40, k_inc: int = 1,
                 l_mult: int = 2, p: float = 0.60, pl: float = 0.99):
        """
        Constructor for the RDPACMethod class.

        Parameters:
            l (int): Size of ranked lists (must be lesser than SIZE_DATASET, this param is mandatory).
            k_start (int): Starting value for k. Default is 1.
            k_end (int): Ending value for k. Default is 40.
            k_inc (int): Increment value for k. Default is 1.
            l_mult (int): Multiplier for l. Default is 2.
            p (float):  Default is 0.60.
            pl (float): Default is 0.99.
        """
        self.method = "RDPAC"
        self.l = l
        self.k_start = k_start
        self.k_end = k_end
        self.k_inc = k_inc
        self.l_mult = l_mult
        self.p = p
        self.pl = pl

    def params(self):
        """
        Returns the parameters of the RDPAC method as a dictionary.
        """
        return {
            "UDL_METHOD": self.method,
            "PARAM_RDPAC_L": self.l,
            "PARAM_RDPAC_K_START": self.k_start,
            "PARAM_RDPAC_K_END": self.k_end,
            "PARAM_RDPAC_K_INC": self.k_inc,
            "PARAM_RDPAC_L_MULT": self.l_mult,
            "PARAM_RDPAC_P": self.p,
            "PARAM_RDPAC_PL": self.pl
        }