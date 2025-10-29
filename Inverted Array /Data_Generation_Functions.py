import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from Classes.File_Management_Classes import Read_Write_Class
from Classes.G2_Classes import *
from Classes.Qutip_Class import *
from Classes.SMUTHI_Environment_Class import *

from enum import Enum


#Enforce consistency in naming conventions
class Approx(Enum): 
    EXACT = "Exact"
    MWISE = "M-wise"
    PAIR = "Pairwise"

#Enforce consistency in naming conventions
class SOCF_Type(Enum): 
    G2 = "G2"
    g2_xx = "g2_xx"
    g2_xy = "g2_xy"


#Generate Emitter Configurations
def Generate_Emitter_Configuration(N, Dim, a_wave):
    Pos = []
    if Dim == 2:
        for i in range(N):
            for j in range(N):
                Pos.append([i*a_wave, j*a_wave, 0])
        return Pos
    elif Dim ==1:
        for i in range(N):
            Pos.append([i*a_wave, 0, 0])
        return Pos

def Generate_Dipole_Moments(N, Dim):
    Dipole_Moments = []
    for i in range(N**Dim):
        Dipole_Moments.append([0,0,1])
    return Dipole_Moments

class Sampling_Methods:

    def __init__(self, SOCF_Type: SOCF_Type, Approximation: Approx, N, Dim, m, omega_0=Frequency_Class(1,"Nat")):
        self.SOCF_Type = SOCF_Type
        self.Approximation = Approximation
        self.N = N
        self.Dim = Dim
        self.m = m

        self.Env = Smuthi_Environment()
        self.wave = Distance_Class(2*np.pi/omega_0.Nat, "Nat")
        self.omega_0 = omega_0

    #Generate Filenames for different configurations
    def Generate_Filename(self, t):
        Filename = self.SOCF_Type.value + "_" + self.Approximation.value + "_N" + str(self.N) + "_Dim" + str(self.Dim)
        Filename = Filename + "_omega" + str(int(self.omega_0.Nat))

        if self.Approximation == Approx.MWISE:
            Filename += "_m" + str(self.m)

        if t == 0:
            Filename += "_t0"
        elif t == np.inf:
            Filename += "_tInf"
        else:
            Filename += "_t" + str(t)

        Filename += ".csv"
        return Filename

    #Generate the Different inverted array datasets
    def Generate_Inverted_Array_Data(self, a_set:np.ndarray, Sampling_Num=100):
        Inverted_Filepath = "Inverted Array /Data Files/Inverted_Array_Data"
        Filename_Str = self.Generate_Filename(0)
        
        Data = Read_Write_Class.Read_From_File(Inverted_Filepath + "/" + Filename_Str)
        seperations = {item[0] for item in Data if item}

        Dipole_Moments = Dipole_Moment_Class(Generate_Dipole_Moments(self.N, self.Dim), "Nat")

        for a in a_set:
            if a not in seperations:
                print(a)
                Emitter_Positions = Generate_Emitter_Configuration(self.N, self.Dim, a)
                G2 = SOCF_Class(self.Env, Emitter_Positions, Dipole_Moments, self.omega_0)
                if self.SOCF_Type == SOCF_Type.G2:
                    G2.Compute_Decay_Rates()
                elif self.SOCF_Type == SOCF_Type.g2_xx:
                    G2.Compute_Far_Field(np.pi/2, np.pi/2, 0, 0)
                elif self.SOCF_Type == SOCF_Type.g2_xy:
                    G2.Compute_Far_Field(np.pi/2, np.pi/2, np.pi/2, 0)

                if self.Approximation == Approx.EXACT:
                    g2 = G2.Exact_SOCF_Inverted(self.SOCF_Type.value)
                elif self.Approximation == Approx.MWISE:
                    g2 = G2.Underestimate_Sampling(self.SOCF_Type.value, Sampling_Num, self.m, 0)
                elif self.Approximation == Approx.PAIR:
                    g2 = G2.Gardiner_Sampling(self.SOCF_Type.value, Sampling_Num, 0)

                Read_Write_Class.Write_To_File(Inverted_Filepath + "/" + Filename_Str, a, g2)





















