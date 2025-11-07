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

#Generate Dipole Moments
def Generate_Dipole_Moments(N, Dim):
    Dipole_Moments = []
    for i in range(N**Dim):
        Dipole_Moments.append([0,0,1])
    return Dipole_Moments


class Sampling_Methods:

    def __init__(self, SOCF_Type: SOCF_Type, Approximation: Approx, N, Dim, m=2, omega_0=Frequency_Class(1,"Nat"), Num_Samples=1000):
        self.SOCF_Type = SOCF_Type
        self.Approximation = Approximation
        self.N = N
        self.Dim = Dim
        self.m = m
        self.Num_Samples = Num_Samples

        self.Env = Smuthi_Environment()
        self.wave = Distance_Class(2*np.pi/omega_0.Nat, "Nat")
        self.omega_0 = omega_0

        self.message = lambda t: f"Computing {SOCF_Type.value}: t={t}, {self.Approximation.value}, N={N**Dim}"


        try:
        # Works when this file is executed directly as a .py
            self.repo_root = Path(__file__).resolve().parents[0]
        except NameError:
            # Works when running from a Jupyter notebook
            self.repo_root = Path().resolve()

    #Generate Filenames for different configurations
    def Generate_Filename(self, t, kL = [1,0,0], Rabi_Factor = 1):
        Filename = self.SOCF_Type.value + "_" + self.Approximation.value + "_N" + str(self.N) + "_Dim" + str(self.Dim)
        Filename = Filename + "_omega" + str(int(self.omega_0.Nat))

        if self.Approximation == Approx.MWISE:
            Filename += "_m" + str(self.m)
            Filename += "_Samples" + str(self.Num_Samples)
        elif self.Approximation == Approx.PAIR:
            Filename += "_Samples" + str(self.Num_Samples)
            
        if t == 0:
            Filename += "_t0"
        elif t == np.inf:
            Filename += "_tInf"
            Filename += "_kL" + str(kL[0]) + str(kL[1]) + str(kL[2])
            Filename += "_Rabi" + str(Rabi_Factor)
        else:
            Filename += "_t" + str(t)

        Filename += ".csv"
        return Filename

    #Generate the Different inverted array datasets
    def Generate_Inverted_Array_Data(self, a_set:np.ndarray):
        Inverted_Filepath = str(self.repo_root / "Data Files" / "Inverted_Array_Data")
        Filename_Str = self.Generate_Filename(0)
        
        Data = Read_Write_Class.Read_From_File(Inverted_Filepath + "/" + Filename_Str)
        seperations = {item[0] for item in Data if item}

        Dipole_Moments = Dipole_Moment_Class(Generate_Dipole_Moments(self.N, self.Dim), "Nat")

        for a in tqdm(a_set, desc=self.message(0), unit="config"):
            if np.round(a,10) not in seperations:
                Emitter_Positions = Distance_Class(Generate_Emitter_Configuration(self.N, self.Dim, a*self.wave.Nat), "Nat")
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
                    g2 = G2.Underestimate_Sampling(self.SOCF_Type.value, self.Num_Samples, self.m, 0)
                elif self.Approximation == Approx.PAIR:
                    g2 = G2.Gardiner_Sampling(self.SOCF_Type.value, self.Num_Samples, 0)

                Read_Write_Class.Write_To_File(Inverted_Filepath + "/" + Filename_Str, a, np.real(g2))
                seperations.add(a)
        return Inverted_Filepath + "/" + Filename_Str

    #Generate Emission data for Exact and M-Wise Approximations
    def Generate_Emission_Data(self, a, t_max, res=501):

        Filename = "Emission_" + self.SOCF_Type.value + "_" + self.Approximation.value + "_N" + str(self.N) + "_Dim" + str(self.Dim)
        Filename = Filename + "_a" + str(np.round(a,10)) + "_omega" + str(int(self.omega_0.Nat)) + "_tmax" + str(t_max) + "_res" + str(res)
        if self.Approximation == Approx.MWISE:
            Filename += "_m" + str(self.m)
            Filename += "_Samples" + str(self.Num_Samples)
        Filename += ".csv"

        Emission_Filepath = str(self.repo_root / "Data Files" / "Emission_Data")

        Positions = Distance_Class(Generate_Emitter_Configuration(self.N, self.Dim, a*self.wave.Nat), "Nat")
        Dipoles = Dipole_Moment_Class(Generate_Dipole_Moments(self.N, self.Dim), "Nat")

        G2 = SOCF_Class(self.Env, Positions, Dipoles, self.omega_0)
        G2.Compute_Decay_Rates()

        if os.path.isfile(Emission_Filepath + "/" + Filename) == False:
            if self.Approximation == Approx.EXACT:
                t, R_Exact = G2.Exact_Emission_Rate(t_max, res)
                for i in range(len(t)):
                    Read_Write_Class.Write_To_File(Emission_Filepath + "/" + Filename, np.real(t[i]), np.real(R_Exact[i]))
            elif self.Approximation == Approx.MWISE:
                t, R_Mwise = G2.Underestimate_Emission_Rate(t_max, self.Num_Samples, self.m, res)
                for i in range(len(t)):
                    Read_Write_Class.Write_To_File(Emission_Filepath + "/" + Filename, np.real(t[i]), np.real(R_Mwise[i]))

        return Emission_Filepath + "/" + Filename

    def Generate_Steady_State_Data(self, a_set:np.ndarray, Rabi_Factor:float=5, kL:Inv_Metre_Class=Inv_Metre_Class([1,0,0], "Nat")):
        Filepath = str(self.repo_root / "Data Files" / "Steady_State_Data")
        Filename = self.Generate_Filename(np.inf, kL.Nat, Rabi_Factor)

        Data = Read_Write_Class.Read_From_File(Filepath + "/" + Filename)
        seperations = {item[0] for item in Data if item}

        Dipole_Moments = Dipole_Moment_Class(Generate_Dipole_Moments(self.N, self.Dim), "Nat")
        Rabi = Frequency_Class(Rabi_Factor * (self.omega_0.Nat)**3 / (3*np.pi), "Nat")

        for a in tqdm(a_set, desc=self.message("inf"), unit="config"):
            if np.round(a,10) not in seperations:
                Emitter_Positions = Distance_Class(Generate_Emitter_Configuration(self.N, self.Dim, a*self.wave.Nat), "Nat")
                G2 = SOCF_Class(self.Env, Emitter_Positions, Dipole_Moments, self.omega_0)
                if self.SOCF_Type == SOCF_Type.G2:
                    G2.Compute_Decay_Rates()
                elif self.SOCF_Type == SOCF_Type.g2_xx:
                    G2.Compute_Far_Field(np.pi/2, np.pi/2, 0, 0)
                elif self.SOCF_Type == SOCF_Type.g2_xy:
                    G2.Compute_Far_Field(np.pi/2, np.pi/2, np.pi/2, 0)

                if self.Approximation == Approx.EXACT:
                    g2 = G2.Exact_SOCF(self.SOCF_Type.value, "inf", Rabi, kL)
                elif self.Approximation == Approx.MWISE:
                    g2 = G2.Underestimate_Sampling(self.SOCF_Type.value, self.Num_Samples, self.m, "inf", Rabi, kL)
                elif self.Approximation == Approx.PAIR:
                    g2 = G2.Gardiner_Sampling(self.SOCF_Type.value, self.Num_Samples, "inf", Rabi, kL)

                Read_Write_Class.Write_To_File(Filepath + "/" + Filename, a, np.real(g2))
                seperations.add(a)
        return Filepath + "/" + Filename

    def Generate_Finite_time_Data(self, a_set:np.ndarray, t):
        Filepath = str(self.repo_root / "Data Files" / "Finite_Time_Data")
        Filename = self.Generate_Filename(t, [0,0,0], 0)

        gamma_0 = (self.omega_0.Nat)**3 / (3*np.pi)

        Data = Read_Write_Class.Read_From_File(Filepath + "/" + Filename)
        seperations = {item[0] for item in Data if item}

        Dipole_Moments = Dipole_Moment_Class(Generate_Dipole_Moments(self.N, self.Dim), "Nat")

        for a in tqdm(a_set, desc=self.message(t), unit="config"):
            if np.round(a,10) not in seperations:
                Emitter_Positions = Distance_Class(Generate_Emitter_Configuration(self.N, self.Dim, a*self.wave.Nat), "Nat")
                G2 = SOCF_Class(self.Env, Emitter_Positions, Dipole_Moments, self.omega_0)
                if self.SOCF_Type == SOCF_Type.G2:
                    G2.Compute_Decay_Rates()
                elif self.SOCF_Type == SOCF_Type.g2_xx:
                    G2.Compute_Far_Field(np.pi/2, np.pi/2, 0, 0)
                elif self.SOCF_Type == SOCF_Type.g2_xy:
                    G2.Compute_Far_Field(np.pi/2, np.pi/2, np.pi/2, 0)

                kL = Inv_Metre_Class([0,0,0], "Nat")
                Rabi = Frequency_Class(0, "Nat")
                if self.Approximation == Approx.EXACT:
                    g2 = G2.Exact_SOCF(self.SOCF_Type.value, t/gamma_0, Rabi, kL)
                elif self.Approximation == Approx.MWISE:
                    g2 = G2.Underestimate_Sampling(self.SOCF_Type.value, self.Num_Samples, self.m, t/gamma_0, Rabi, kL)
                elif self.Approximation == Approx.PAIR:
                    g2 = G2.Gardiner_Sampling(self.SOCF_Type.value, self.Num_Samples, t/gamma_0, Rabi, kL)

                Read_Write_Class.Write_To_File(Filepath + "/" + Filename, a, np.real(g2))
                seperations.add(a)
        return Filepath + "/" + Filename

    def Inverted_Array_Distributions(self, a, N_Samples):
        Filepath = str(self.repo_root / "Data Files" / "Distributions")
        Filename = f"Mwise_Dist_m{self.m}_a{a}_N{self.N}_Dim{self.Dim}.txt"

        Data = Read_Write_Class.Read_In_1D_Data(Filepath + "/" + Filename) 
        if len(Data) == 0 and N_Samples == 0:
            raise Exception("The current configuration will produce no data point to plot")
        else:

            Dipole_Moments = Dipole_Moment_Class(Generate_Dipole_Moments(self.N, self.Dim), "Nat")
            Emitter_Positions = Distance_Class(Generate_Emitter_Configuration(self.N, self.Dim, a*self.wave.Nat), "Nat")
            G2 = SOCF_Class(self.Env, Emitter_Positions, Dipole_Moments, self.omega_0)
            if self.SOCF_Type == SOCF_Type.G2:
                G2.Compute_Decay_Rates()
            elif self.SOCF_Type == SOCF_Type.g2_xx:
                G2.Compute_Far_Field(np.pi/2, np.pi/2, 0, 0)
            elif self.SOCF_Type == SOCF_Type.g2_xy:
                G2.Compute_Far_Field(np.pi/2, np.pi/2, np.pi/2, 0)

            for S in range(N_Samples):
                g2 = G2.Underestimate_Sampling(self.SOCF_Type.value, 1, self.m, 0)
                Read_Write_Class.Append_to_1D_Data(Filepath + "/" + Filename, np.real(g2))

        return Read_Write_Class.Read_In_1D_Data(Filepath + "/" + Filename)


                
        




