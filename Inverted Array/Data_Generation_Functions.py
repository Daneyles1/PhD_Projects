import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from Classes.File_Management_Classes import Read_Write_Class
from Classes.G2_Classes import *
from Classes.Qutip_Class import *
from Classes.SMUTHI_Environment_Class import *

from tqdm import tqdm
from enum import Enum
import os

class BIC_Type(Enum):
    MDBIC = [[0,1,0], 0.163*400, 708.9, "MDBIC"]
    EDBIC = [[0,0,1], 0, 552, "EDBIC"]
    FS = [[0,1,0], 0.163*400, 708.9, "FS"]

class Inverted_Array:

    z0 = 104

    def __init__(self, BIC:BIC_Type, N_Sca:int = 0, mult=1):
        self.dip = BIC.value[0]
        self.x0 = BIC.value[1]
        self.wave = Distance_Class(BIC.value[2], "Nano")
        self.omega_0 = Frequency_Class(2*np.pi/self.wave.Nat, "Nat")
        self.Name = BIC.value[3]

        self.mult = 0
        self.N_Sca = 0

        self.Env = Smuthi_Environment()
        self.Env.show = False
        if self.Name != "FS":
            self.N_Sca = N_Sca
            self.mult = mult
            self.Env.Generate_Diego_ED_BIC(mult, N_Sca)

        try:
        # Works when this file is executed directly as a .py
            self.repo_root = Path(__file__).resolve().parents[0]
        except NameError:
            # Works when running from a Jupyter notebook
            self.repo_root = Path().resolve()


    def __Generate_QE_Array(self, N_Row, a):
        R = []
        N = int(np.floor(N_Row/2))
        for i in range(-N, N+1):
            for j in range(-N, N+1):
                R.append([self.x0+a*i, a*j, self.z0])
        return R

    def __Generate_Dipole_Set(self, N_Row):
        Dips = []
        for i in range(N_Row**2):
            Dips.append(self.dip)
        return Dips
    
    def Compute_Emission_Data(self, N_Row:int, a:Distance_Class, t_max, res=200):
        Filepath = str(self.repo_root / "Data Files" / "Emission Data")
        Filename = f"Emission_{self.Name}_NQE{N_Row}_a{a.Nano}_NSca{self.N_Sca}_m{self.mult}_t{t_max}_r{res}.csv"
        File = Filepath + "/" + Filename

        if Read_Write_Class.Check_File_Exists(File) == True:
            return File
        else:
            Pos = Distance_Class(self.__Generate_QE_Array(N_Row, a.Nano), "Nano")
            Dips = Dipole_Moment_Class(self.__Generate_Dipole_Set(N_Row), "Nat")

            Gamma, Delta = self.Env.Calculate_Rate_Matrices(Pos, Dips, self.wave)
            Qut = Qutip_Solver(Gamma, Delta, self.omega_0)
            t, R = Qut.Emission_Rate_over_time(t_max, res, normalize=True)
            for i in range(len(res)):
                Read_Write_Class.Write_To_File(File, t[i], R[i])
            return File



