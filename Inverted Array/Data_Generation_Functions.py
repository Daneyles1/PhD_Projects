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

class BIC_Type(Enum): #Class summarisiong the different possible configurations of the BIC and FS
    MDBIC = [[0,1,0], 0.163*400, 708.9, "MDBIC"] #Dipole moment, x0 offset, wavelength(nm), name
    EDBIC = [[0,0,1], 0, 552, "EDBIC"]
    FS_MD = [[0,1,0], 0.163*400, 708.9, "FS_MD"]
    FS_ED = [[0,0,1], 0, 552, "FS_ED"]

class Inverted_Array:

    z0 = 104 #Z offset of dipole to be above the metasurface

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
        if self.Name[:2] != "FS": #If its not FS, then generate the metasurface
            self.N_Sca = N_Sca
            self.mult = mult
            self.Env.Generate_Diego_ED_BIC(mult, N_Sca)

        try:
        # Works when this file is executed directly as a .py
            self.repo_root = Path(__file__).resolve().parents[0]
        except NameError:
            # Works when running from a Jupyter notebook
            self.repo_root = Path().resolve()

    #Generate 2D array of emitter positions centres on (x0, 0, z0) in x-y plane
    def __Generate_QE_Array(self, N_Row, a):
        R = []
        N = int(np.floor(N_Row/2))
        for i in range(-N, N+1):
            for j in range(-N, N+1):
                R.append([self.x0 + a*i, a*j, self.z0])
        return R

    #Generate N_Row^2 dipole emitters 
    def __Generate_Dipole_Set(self, N_Row):
        Dips = []
        for i in range(N_Row**2):
            Dips.append(self.dip)
        return Dips
    
    #Compute the Emission Rate data for a N_Row^2 of emitters up to t_max*gamma_11
    def Compute_Emission_Data(self, N_Row:int, a:Distance_Class, t_max, res=200):
        Filepath = str(self.repo_root / "Data Files" / "Emission Data")
        Filename = f"Emission_{self.Name}_NQE{N_Row}_a{a.Nano}_NSca{self.N_Sca}_m{self.mult}_t{t_max}_r{res}_"
        Filename = Filename + f"d{self.dip[0]}{self.dip[1]}{self.dip[2]}.csv"
        File = Filepath + "/" + Filename

        if Read_Write_Class.Check_File_Exists(File) == True:
            return File
        else:
            Pos = Distance_Class(self.__Generate_QE_Array(N_Row, a.Nano), "Nano")
            Dips = Dipole_Moment_Class(self.__Generate_Dipole_Set(N_Row), "Nat")

            Gamma, Delta = self.Env.Calculate_Rate_Matrices(Pos, Dips, self.wave)
            Qut = Qutip_Solver(Gamma, Delta, self.omega_0)
            t, R = Qut.Emission_Rate_over_time(t_max, res, normalize=True)
            for i in range(res):
                Read_Write_Class.Write_To_File(File, t[i], R[i])
            return File

    #Generate purcell factor for a QE optimally positioned and oriented 
    def Generate_Purcell_Data(self, wave_set:Distance_Class):
        Filepath = str(self.repo_root / "Data Files" / "Purcell Factors")
        Filename = f"Emission_{self.Name}_NSca{self.N_Sca}_m{self.mult}_d{self.dip[0]}{self.dip[1]}{self.dip[2]}.csv"
        File = Filepath + "/" + Filename

        Data = Read_Write_Class.Read_From_File(File)
        waves_Done = {item[0] for item in Data if item}

        Pos = Distance_Class([[self.x0, 0, self.z0],[self.x0, 0, self.z0]], "Nano")
        Dip = Dipole_Moment_Class([self.dip, self.dip], "Nat")

        for wave in tqdm(wave_set.Nano, desc=f"Purcell Factor, N_Sca={self.N_Sca}", unit="config"):
            if np.round(wave,10) not in waves_Done:
                wave =  Distance_Class(wave,"Nano")
                Gamma, Delta = self.Env.Calculate_Rate_Component(Pos, Dip, wave)
                Gamma_0 = np.linalg.norm(self.dip) * ((2*np.pi/wave.Nat)**3) / (3*np.pi)
                Purcell = Gamma.Nat / Gamma_0
                Read_Write_Class.Write_To_File(File, np.round(wave.Nano,10), Purcell)
                waves_Done.add(wave)
        return File

    #Compute and save the Decay matrix of a larger Lattice of emitters
    def __Compute_Full_Decay_Matrix(self, N_Row_QE, a=Distance_Class(400,"Nano")):
        Filepath = str(self.repo_root / "Data Files" / "Gamma Matrices")
        Filename = f"Matrix_{self.Name}_NSca{self.N_Sca}_m{self.mult}_d{self.dip[0]}{self.dip[1]}{self.dip[2]}_NQE{N_Row_QE}_a{a.Nano}.csv"
        File = Filepath + "/" + Filename

        if Read_Write_Class.Check_File_Exists(File) == False:
            Prev_Filename = lambda N: f"Matrix_{self.Name}_NSca{self.N_Sca}_m{self.mult}_d{self.dip[0]}{self.dip[1]}{self.dip[2]}_NQE{N}_a{a.Nano}.csv"
            Alternate_Runs = [i for i in range(3, 21) if i%2 == 1]
            Best_Option = -1 #Use previous run to populate matrix to speed up calculation
            for N in Alternate_Runs:
                if Read_Write_Class.Check_File_Exists(Filepath+"/"+Prev_Filename(N)) == True:
                    Best_Option = N
            

            if Best_Option != -1:
                Previous_Gamma_Matrix = np.zeros([N_Row_QE**2, N_Row_QE**2])
                Previous_Positions = self.__Generate_QE_Array(Best_Option, a.Nano)
                if Best_Option != 1:
                    Previous_Gamma_Matrix = Read_Write_Class.Read_In_2D_Data(Filepath+"/"+Prev_Filename(Best_Option))

                Positions = self.__Generate_QE_Array(N_Row_QE, a.Nano)
                Gamma_Matrix = np.zeros([N_Row_QE**2, N_Row_QE**2])
            
                for i in tqdm(range(N_Row_QE**2), desc="Autofilling Matrix", unit="config"):
                    for j in range(N_Row_QE**2):
                        New_Pos1 = Positions[i]
                        New_Pos2 = Positions[j]

                        for mu in range(Best_Option**2):
                            for nu in range(Best_Option**2):
                                Old_Pos1 = Previous_Positions[mu]
                                Old_Pos2 = Previous_Positions[nu]
                                if np.allclose(New_Pos1, Old_Pos1, atol=1e-12) == True:
                                    if np.allclose(New_Pos2, Old_Pos2, atol=1e-12) == True:
                                        Gamma_Matrix[i][j] = Previous_Gamma_Matrix[mu][nu]
                                        Gamma_Matrix[j][i] = Previous_Gamma_Matrix[mu][nu]
                                        
                                        Read_Write_Class.Write_2D_Data(File, Gamma_Matrix)

        Positions = self.__Generate_QE_Array(N_Row_QE, a.Nano) #Compute remainining empty values
        Gamma_Matrix = Read_Write_Class.Read_In_2D_Data(File)
        if len(Gamma_Matrix) == 0:
            Gamma_Matrix = np.zeros([N_Row_QE**2, N_Row_QE**2])
        Dips = Dipole_Moment_Class([self.dip, self.dip], "Nat")
        for i in tqdm(range(N_Row_QE**2), desc="Computing Gamma Matrix", unit="config"):
            for j in range(N_Row_QE**2):
                if Gamma_Matrix[i][j] == 0:
                    Pos = Distance_Class([Positions[i], Positions[j]], "Nano")
                    Gamma, Delta = self.Env.Calculate_Rate_Component(Pos, Dips, self.wave)
                    Gamma_Matrix[i][j] = Gamma.Nat
                    Read_Write_Class.Write_2D_Data(File, Gamma_Matrix)

        return File

    #Compute G2 as a function of the number of emitters in the 2D array
    def Compute_G2_Vs_NQE(self, N_Row_Max_QE, a=Distance_Class(400,"Nano")):
        Filepath = str(self.repo_root / "Data Files" / "G2_V_N")
        Filename = f"G2_V_n_{self.Name}_NSca{self.N_Sca}_m{self.mult}_d{self.dip[0]}{self.dip[1]}{self.dip[2]}_a{a.Nano}.csv"
        File = Filepath + "/" + Filename

        Data = Read_Write_Class.Read_From_File(File)
        N_Done = {item[0] for item in Data if item}

        N_Row_Set = [i for i in range(N_Row_Max_QE+1) if i %2 ==1]

        for N in N_Row_Set:
            if N not in N_Done:
                Gamma_File = self.__Compute_Full_Decay_Matrix(N, a)
                Gamma = np.array(Read_Write_Class.Read_In_2D_Data(Gamma_File))
                G2 = np.real(Emission_Class.SOCF_Inverted_Calculation(Gamma, Gamma))
                Read_Write_Class.Write_To_File(File, np.round(N,10), np.round(G2,10))
        return File

    #Compute G2 as a function of the emitter array constant (d) i.e. d=0 means all emitters at (x0, 0, z0)
    def Compute_G2_Vs_d(self, d_Ratio_Set, N_Row_QE):
        Filepath = str(self.repo_root / "Data Files" / "G2 seperation Data")
        Filename = f"G2_V_d_{self.Name}_NRowQE{N_Row_QE}_NSca{self.N_Sca}_m{self.mult}_d{self.dip[0]}{self.dip[1]}{self.dip[2]}.csv"
        File = Filepath + "/" + Filename

        Data = Read_Write_Class.Read_From_File(File)
        d_done = {item[0] for item in Data if item}

        Dips = Dipole_Moment_Class(self.__Generate_Dipole_Set(N_Row_QE), "Nat")
        for d in tqdm(d_Ratio_Set, desc="Computing G2", unit='congig'):
            d = np.round(d,10)
            if d not in d_done:
                Pos = Distance_Class(self.__Generate_QE_Array(N_Row_QE, d*self.wave.Nano), "Nano")
                G2 = SOCF_Class(self.Env, Pos, Dips, self.omega_0)
                G2.Compute_Decay_Rates()
                g2 = G2.Exact_SOCF_Inverted("G2")
                Read_Write_Class.Write_To_File(File, np.round(d, 10), np.round(g2, 10))
        return File











































