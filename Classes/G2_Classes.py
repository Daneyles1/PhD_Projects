from enum import Enum
from pathlib import Path
import sys

from .Unit_Converter_Classes import *
from .SMUTHI_Environment_Class import *
from .Qutip_Class import *

import numpy as np
import random
from qutip import *
import scipy.linalg as la


class Emission_Class:

    @staticmethod
    def Kron_Delta(i, j):
        a = 0
        if i == j:
            a = 1
        return a


#Correlation Function Computation   
    @staticmethod
    def SOCF_State_Calculation_Fixed_State(G1_mat, G2_mat, Matrix1, Matrix2): #Compute A given Correlation Funcion
        N = len(Matrix1)
        G1_1 = 0
        for i in range(N):
            for j in range(N):
                G1_1 = G1_1 + Matrix1[j][i] * G1_mat[i][j]
        G1_2 = 0
        for i in range(N):
            for j in range(N):
                G1_2 = G1_2 + Matrix2[j][i] * G1_mat[i][j]
        G2 = 0
        for i in range(N):
            for j in range(N):
                    for k in range(N):
                        for l in range(N):
                            G2 = G2 + Matrix1[l][i] * Matrix2[k][j] * G2_mat[i][j][k][l]
        return G2/(G1_1 * G1_2)
    
    @staticmethod
    def SOCF_State_Calculation(State, Matrix1, Matrix2, Sp, Sm):
        G2 = 0
        N = len(Matrix1)
        for mu in range(N):
            for nu in range(N):
                for gamma in range(N):
                    for epsilon in range(N):
                        G2 = G2 + Matrix1[epsilon][mu] * Matrix2[gamma][nu] * expect(Sp[mu]*Sp[nu]*Sm[gamma]*Sm[epsilon], State)
        G1_a = 0
        G1_b = 0
        for mu in range(N):
            for nu in range(N):
                exp = expect(Sp[mu]*Sm[nu], State)
                G1_a = G1_a + Matrix1[nu][mu] * exp
                G1_b = G1_b + Matrix2[nu][mu] * exp
        return G2/(G1_a*G1_b)

    @staticmethod
    def SOCF_Inverted_Calculation(Matrix1, Matrix2):
        N = len(Matrix1)
        G1_1 = np.trace(Matrix1)
        G1_2 = np.trace(Matrix2)

        matmul = Matrix1*Matrix2

        term1 = np.sum(np.diag(Matrix1)) * np.sum(np.diag(Matrix2))
        term2 = np.sum(matmul.flatten())
        term3 = -2* np.sum(np.diag(matmul).flatten())

        G2 = term1 + term2 + term3
        return G2/(G1_1 * G1_2)
    
    @staticmethod
    def SOCF_Inverted_Calculation_Old(Matrix1, Matrix2):
        N = len(Matrix1)
        G1_1 = np.trace(Matrix1)
        G1_2 = np.trace(Matrix2)
        G2 = 0
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    for l in range(N):
                        delta_term_1 = (1 - Emission_Class.Kron_Delta(i,j)) * (1 - Emission_Class.Kron_Delta(k,l))
                        delta_term_2 = (Emission_Class.Kron_Delta(i,k)*Emission_Class.Kron_Delta(j,l) + Emission_Class.Kron_Delta(i,l)*Emission_Class.Kron_Delta(j,k)) 
                        G2 = G2 + Matrix1[l][i] * Matrix2[k][j] * delta_term_1 * delta_term_2
        return G2/(G1_1 * G1_2)


#Inverted Array Method
    @staticmethod
    def Garcia_Homogenious_Calculation(Matrix): 
        N = len(Matrix)
        EigenValues = np.linalg.eigvals(Matrix)
        Variance = np.var(EigenValues/np.mean(EigenValues), dtype="complex_")
        return ((N-1)/N) + (1/N) * Variance

    @staticmethod
    def Garcia_Calculation_Generalised(Matrix):
        N = len(Matrix)
        eigenvalues, eigenvectors = np.linalg.eigh(Matrix)
        Sum = np.sum(eigenvalues)
        Squared_Sum = np.sum(eigenvalues*eigenvalues)
        term_1 = 1
        term_2 = Squared_Sum/(Sum**2)

        term3_temp = 0
        for i in range(N):
            temp = 0
            for mu in range(N):
                temp = temp + eigenvalues[mu] * eigenvectors[:,mu][i] * np.conjugate(eigenvectors[:,mu][i])
            term3_temp = term3_temp + temp**2
        term_3 = -2*term3_temp/(Sum**2)

        return term_1 + term_2 + term_3
    
    @staticmethod
    def Inverted_Array_G_Matrices(N):
        G1_Mat = np.zeros([N, N], dtype="complex_")
        G2_Mat = np.zeros([N, N, N, N], dtype="complex_")
        for i in range(N):
            G1_Mat[i][i] = 1
        
        for i in range(N):
            for j in range(N):
                if i !=j:
                    for k in range(N):
                        for l in range(N):
                            if k != l:
                                delta_term_2 = (Emission_Class.Kron_Delta(i,k)*Emission_Class.Kron_Delta(j,l) + Emission_Class.Kron_Delta(i,l)*Emission_Class.Kron_Delta(j,k)) 
                                G2_Mat[i][j][k][l] = delta_term_2
        return G1_Mat, G2_Mat

    @staticmethod
    def G3_Inverted_Garcia(Matrix):
        N = len(Matrix)
        EigenValues = np.linalg.eigvals(Matrix)
        Eigen = EigenValues/(N*np.mean(EigenValues))
        return 1 + 2*np.sum(Eigen**3) + (3-(12/N))*np.sum(Eigen**2) + (12/(N**2)) - (6/N)


#SubSet Generation
    @staticmethod
    def Generate_Subset_List(N, Num_of_Samples, Sample_Size): #Generate subsets for Gardiner or Underestimate Methods
        Samples = []
        while len(Samples) < Num_of_Samples:
            Samples.append(random.sample(range(0, N), Sample_Size))
        return Samples
    
    @staticmethod
    def Generate_Subset_Nearest_Neighbour(Positions:Distance_Class, Sample_Size, Overlap:bool): #Generate Subsets for Nearest Neighbour Methods TRUE FOR OVERLAP
        N = len(Positions)
        indexes = np.arange(0, N)
        Used = np.full(N, False)
        Samples = []

        for index in range(N):
            distances = []
            if Used[index] == False:
                for i in range(N):
                    dist_Vec = np.array(Positions.Nat[i]) - np.array(Positions.Nat[index])
                    distances.append(np.linalg.norm(dist_Vec))
                distance, indexe = (list(t) for t in zip(*sorted(zip(distances, indexes))))
                Nearest_Neighbors = []
                for i in indexe:
                    if Used[i] == False and len(Nearest_Neighbors) < Sample_Size:
                        Nearest_Neighbors.append(i)
                        if Overlap == False:
                            Used[i] = True
                if len(Nearest_Neighbors) == Sample_Size:
                    Samples.append(Nearest_Neighbors)
        return Samples

    @staticmethod
    def Generate_Subset_Matrix(subset, Matrix): #Generate the needed matrices for a given Subset of indexes
        Matrix_SubSet = np.zeros([len(subset),len(subset)], dtype='complex_')
        for i in range(len(subset)):
            for j in range(len(subset)):
                Matrix_SubSet[i][j] = Matrix[subset[i]][subset[j]]
        return Matrix_SubSet

    @staticmethod
    def Generate_Subset_of_a_List(subset, List):
        Temp_SubList = []
        for i in range(len(subset)):
            Temp_SubList.append(List[subset[i]])
        return Temp_SubList

    @staticmethod
    def Gardiner_Approximation(state:Qobj, Matrix_1, Matrix_2):
        SigmaMinus_Gardiner = [tensor(identity(2), sigmam()), tensor(sigmam(), identity(2))]
        SigmaPlus_Gardiner = [tensor(identity(2), sigmap()), tensor(sigmap(), identity(2))]
        g1 = 0
        for mu in range(2):
            for nu in range(2):
                term1 =  Matrix_2[nu][mu] * expect(SigmaPlus_Gardiner[mu]*SigmaMinus_Gardiner[nu], state)
                for gamma in range(2):
                    for epsilon in range(2):
                        if mu == nu and mu == gamma and nu == epsilon:
                            pass
                        else:
                            term2 = Matrix_1[epsilon][gamma] * expect(SigmaPlus_Gardiner[gamma] * SigmaMinus_Gardiner[epsilon], state)
                            g1 = g1 + term1 * term2
        g2 = 0
        for mu in range(2):
            for nu in range(2):
                for gamma in range(2):
                    for epsilon in range(2):
                        if mu == nu and mu == gamma and nu == epsilon:
                            pass
                        else:
                            Op = SigmaPlus_Gardiner[mu] * SigmaPlus_Gardiner[nu] * SigmaMinus_Gardiner[gamma] * SigmaMinus_Gardiner[epsilon]
                            g2 = g2 + Matrix_1[epsilon][mu] * Matrix_2[gamma][nu] * expect(Op, state)
        return g2/g1


#Analytical Values
    @staticmethod
    def Gamma_0_FS(omega0, d):
        return Frequency_Class(((omega0**3)/(3*np.pi)) * np.dot(np.conj(d), d), "Nat")

    @staticmethod
    def Greens_Fs(omega0:Frequency_Class, R1, R2):#Analytical Free Space Greens Function
        G = np.zeros([3,3])
        R = np.real(np.linalg.norm(np.array(R1) - np.array(R2)))
        if R == 0:
            G = (2j/3)*(omega0.Nat/(4*np.pi)) * np.identity(3)
        else:
            Rhat = np.real((np.array(R1) - np.array(R2))/R)
            RR = np.zeros([3,3])
            Id = np.identity(3)
            for i in range(3):
                for j in range(3):
                    RR[i][j] = np.real(Rhat[i]*Rhat[j])
            g = (1/(4*np.pi*R)) * np.exp(1j*omega0.Nat * R) 
            term_1 = (3/((omega0.Nat*R)**2)) - (3j/(omega0.Nat*R)) -1
            term_2 = 1 + 1j/(omega0.Nat*R) - 1/((omega0.Nat*R)**2)
            G = (term_1*RR + term_2*Id)*g 
        return G
    
    @staticmethod
    def Calculate_Rate_Matrices_FS(Positions:Distance_Class, Moments:Dipole_Moment_Class, omega0:Frequency_Class):
        N = len(Positions.Nat)
        if omega0.Nat >0:
            gamma = np.zeros([N, N])
            delta = np.zeros([N, N])
            for i in range(N):
                for j in range(N):
                    pos_i = Positions.Nat[i] 
                    pos_j = Positions.Nat[j] 
                    dip_i = Moments.Nat[i]
                    dip_j = Moments.Nat[j]
                    if j > i:
                        G = Emission_Class.Greens_Fs(omega0, pos_i, pos_j)
                        ImG_ij =  np.imag(np.matmul(np.conj(dip_i), np.matmul(G, dip_j)))
                        ReG_ij =  np.real(np.matmul(np.conj(dip_i), np.matmul(G, dip_j)))
                        gamma[i][j] = (2 * (omega0.Nat**2)) * ImG_ij
                        gamma[j][i] = np.conj(gamma[i][j])
                        delta[i][j] = -(omega0.Nat**2) * ReG_ij
                        delta[j][i] = np.conj(delta[i][j])
                    if i == j:
                        gamma[i][j] = np.real(np.dot(Moments.Nat[i],np.conj(Moments.Nat[i])) * (omega0.Nat**3)/(3*np.pi))
            return Frequency_Class(gamma, "nat"), Frequency_Class(delta, "nat")
        else:
            return Frequency_Class(np.zeros([N,N]), "nat"), Frequency_Class(np.zeros([N,N]), "nat")

    @staticmethod
    def Calculate_Far_Field_Matrix_FS(Positions:Distance_Class, Moments:Dipole_Moment_Class, omega0:Frequency_Class, Polar1, Polar2, Azim1, Azim2):
        N = len(Positions.Nano)
        Det_Dir_1 = [np.sin(Polar1)*np.cos(Azim1), np.sin(Polar1)*np.sin(Azim1), np.cos(Polar1)]
        Det_Dir_2 = [np.sin(Polar2)*np.cos(Azim2), np.sin(Polar2)*np.sin(Azim2), np.cos(Polar2)]
        Det_1 = Distance_Class((10**7) * np.array(Det_Dir_1), "nat")
        Det_2 = Distance_Class((10**7) * np.array(Det_Dir_2), "nat")
        Far_1 = np.zeros([N, N], dtype="complex_")
        Far_2 = np.zeros([N, N], dtype="complex_")
        for i in range(N):
            for j in range(N):
                pos_1 = Positions.Nat[i]
                pos_2 = Positions.Nat[j]
                Green1 = Emission_Class.Greens_Fs(omega0, pos_1, Det_1.Nat)
                Green2 = Emission_Class.Greens_Fs(omega0, pos_2, Det_1.Nat)
                vec_1 = np.dot(Green1, Moments.Nat[i])
                vec_2 = np.dot(Green2, Moments.Nat[j])
                Far_1[i][j] = np.dot(vec_1, np.conj(vec_2))

                Green1 = Emission_Class.Greens_Fs(omega0, pos_1, Det_2.Nat)
                Green2 = Emission_Class.Greens_Fs(omega0, pos_2, Det_2.Nat)
                vec_1 = np.dot(Green1, Moments.Nat[i])
                vec_2 = np.dot(Green2, Moments.Nat[j])
                Far_2[i][j] = np.dot(vec_1, np.conj(vec_2))
        return Far_1, Far_2

class SOCF_Class:
    Same_Far = False
    Force_SMUTHI = False

    @staticmethod
    def How_To_Use():
        print("""This class containes the needed functions for computing the second order correlation function
            - One of the following functions must be run to enable further computations
                - To consider G2, run "Compute_Decay_Rates"
                - To consider g2, run "Compute_Far_Field"
            - Run either the "Undersetimate_" or "Gardiner_" functions to compute the SOCF using the sampling methods
            - Run Exact to compute true SOCF value (Ensure number of emitters is small)
            - "Fixed_State" allows iterative calculations for the same state (for, say, varying the angle of emission in g2)
        """)

#First Use functions
    def __init__(self, Env:Smuthi_Environment, Positions:Distance_Class, Moments:Dipole_Moment_Class, omega0:Frequency_Class):
        self.Env = Env
        self.Positions = Positions
        self.Moments = Moments
        self.omega0 = omega0
        self.wavelength = Distance_Class(2*np.pi/omega0.Nat, "nat")
    
    def Compute_Decay_Rates(self, Gamma_in:Frequency_Class=Frequency_Class([], "SI"), Delta_in:Frequency_Class=Frequency_Class([], "SI")):
        Gamma = None
        Delta = None
        if self.Env.MetaSurface == "FS" and self.Force_SMUTHI == False:
            Gamma, Delta = Emission_Class.Calculate_Rate_Matrices_FS(self.Positions, self.Moments, self.omega0)
        else:
            Gamma, Delta = self.Env.Calculate_Rate_Matrices(self.Positions, self.Moments, self.wavelength, Gamma_in, Delta_in)
        self.Gamma_Matrix = Gamma
        self.Delta_Matrix = Delta
    
    def Compute_Far_Field(self, Polar1, Polar2, Azim1, Azim2):
        if self.Env.MetaSurface == "FS":
            Far_1, Far_2 = Emission_Class.Calculate_Far_Field_Matrix_FS(self.Positions, self.Moments, self.omega0, Polar1, Polar2, Azim1, Azim2)
        else:
            Far_1, Far_2 = self.Env.Calculate_Greens_Far_Field_Far(self.Positions, self.Moments, self.wavelength, Polar1, Polar2, Azim1, Azim2)
        if Polar1 == Polar2 and Azim1 == Azim2:    
            self.Same_Far = True
        self.Far_1_Matrix = Far_1
        self.Far_2_Matrix = Far_2

#Inverted Calculations
    def Exact_SOCF_Inverted(self, SOCF_Type):
        G2 = None
        if SOCF_Type[:2] == "G2":
            G2 = Emission_Class.SOCF_Inverted_Calculation(self.Gamma_Matrix.Nat, self.Gamma_Matrix.Nat)
            #G2 = Emission_Class.SOCF_Inverted_Calculation(self.Gamma_Matrix.Nat, self.Gamma_Matrix.Nat)
        elif SOCF_Type[:2] == "g2":
            if self.Same_Far == True:
                G2 = Emission_Class.SOCF_Inverted_Calculation(self.Far_1_Matrix, self.Far_1_Matrix)
            else:
                G2 = Emission_Class.SOCF_Inverted_Calculation(self.Far_1_Matrix, self.Far_2_Matrix)
        return G2

    def Compute_G_Inverted(self, Matrix1, Matrix2):
        G2 = 0
        N = len(self.Positions.Nat)
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    for l in range(N):
                        delta_term_1 = (1 - Emission_Class.Kron_Delta(i,j)) * (1 - Emission_Class.Kron_Delta(k,l))
                        delta_term_2 = (Emission_Class.Kron_Delta(i,k)*Emission_Class.Kron_Delta(j,l) + Emission_Class.Kron_Delta(i,l)*Emission_Class.Kron_Delta(j,k)) 
                        G2 = G2 + Matrix1[l][i] * Matrix2[k][j] * delta_term_1 * delta_term_2
        return G2


#Evolved System
    def Exact_SOCF_Fixed_State(self, SOCF_Type, G2_Mat, G1_Mat):
        G2 = None
        if SOCF_Type[:2] == "G2":
            G2 = Emission_Class.SOCF_State_Calculation_Fixed_State(G1_Mat, G2_Mat, self.Gamma_Matrix.Nat, self.Gamma_Matrix.Nat)
        elif SOCF_Type[:2] == "g2":
            G2 = Emission_Class.SOCF_State_Calculation_Fixed_State(G1_Mat, G2_Mat, self.Far_1_Matrix, self.Far_2_Matrix)
        return G2

    def Exact_SOCF(self, SOCF_Type, t, Rabi:Frequency_Class, kL:Inv_Metre_Class):
        QutiP = Qutip_Solver(self.Positions, self.Gamma_Matrix, self.Delta_Matrix, self.omega0, Rabi, kL)
        Sp = Qutip_Solver.Generate_m_Operators(len(self.Positions.Nat), qutip.sigmap())
        Sm = Qutip_Solver.Generate_m_Operators(len(self.Positions.Nat), qutip.sigmam())
        state = None
        if t == "inf" or t == "Inf":
            state = QutiP.Solve_Steady_State()
        else:
            state = QutiP.Calculate_State_at_t(t)

        G2 = None
        if SOCF_Type[:2] == "G2":
            G2 = Emission_Class.SOCF_State_Calculation(state, self.Gamma_Matrix.Nat, self.Gamma_Matrix.Nat, Sp, Sm)
        elif SOCF_Type[:2] == "g2":
            G2 = Emission_Class.SOCF_State_Calculation(state, self.Far_1_Matrix, self.Far_2_Matrix, Sp, Sm)
        return G2
        
    def Underestimate_Sampling(self, SOCF_Type:str, Num_of_Samples:int, Sample_Size:int, t, Rabi:Frequency_Class=None, kL:Inv_Metre_Class=None):
        Samples = Emission_Class.Generate_Subset_List(len(self.Positions), Num_of_Samples, Sample_Size)
        Sp = Qutip_Solver.Generate_m_Operators(len(self.Positions.Nat), qutip.sigmap())
        Sm = Qutip_Solver.Generate_m_Operators(len(self.Positions.Nat), qutip.sigmam())
        G2_Samples = []
        if t == 0:
            if SOCF_Type[:2] == "G2":
                for sample in Samples:
                    Gamma_Matrix_Subset = Emission_Class.Generate_Subset_Matrix(sample, self.Gamma_Matrix.Nat)
                    G2_Samples.append(Emission_Class.Garcia_Homogenious_Calculation(Gamma_Matrix_Subset))
            elif SOCF_Type[:2] == "g2":
                if self.Same_Far == False:
                    for sample in Samples:
                        Green_Far_1_SubSet = Emission_Class.Generate_Subset_Matrix(sample, self.Far_1_Matrix)
                        Green_Far_2_SubSet = Emission_Class.Generate_Subset_Matrix(sample, self.Far_2_Matrix)
                        G2_Samples.append(Emission_Class.SOCF_Inverted_Calculation(Green_Far_1_SubSet, Green_Far_2_SubSet))
                else:
                    for sample in Samples:
                        Green_Far_1_SubSet = Emission_Class.Generate_Subset_Matrix(sample, self.Far_1_Matrix)
                        G2_Samples.append(Emission_Class.Garcia_Homogenious_Calculation(Green_Far_1_SubSet))
        if t == "inf":
            if SOCF_Type[:2] == "G2":
                for sample in Samples:
                    Gamma_Matrix_Subset = Frequency_Class(Emission_Class.Generate_Subset_Matrix(sample, self.Gamma_Matrix.Nat), "Nat")
                    Delta_Matrix_Subset = Frequency_Class(Emission_Class.Generate_Subset_Matrix(sample, self.Delta_Matrix.Nat), "Nat")
                    Positions_Subset = Distance_Class(Emission_Class.Generate_Subset_of_a_List(sample, self.Positions.Nat), "Nat")
                    Qut = Qutip_Solver(Positions_Subset, Gamma_Matrix_Subset, Delta_Matrix_Subset, self.omega0, Rabi, kL)
                    state = Qut.Solve_Steady_State()
                    G2_Samples.append(Emission_Class.SOCF_State_Calculation(state, Gamma_Matrix_Subset.Nat, Gamma_Matrix_Subset.Nat, Sp, Sm))
            elif SOCF_Type[:2] == "g2":
                for sample in Samples:
                    Gamma_Matrix_Subset = Frequency_Class(Emission_Class.Generate_Subset_Matrix(sample, self.Gamma_Matrix.Nat), "Nat")
                    Delta_Matrix_Subset = Frequency_Class(Emission_Class.Generate_Subset_Matrix(sample, self.Delta_Matrix.Nat), "Nat")
                    Green_Far_1_SubSet = Emission_Class.Generate_Subset_Matrix(sample, self.Far_1_Matrix)
                    Green_Far_2_SubSet = Emission_Class.Generate_Subset_Matrix(sample, self.Far_2_Matrix)
                    Qut = Qutip_Solver(Positions_Subset, Gamma_Matrix_Subset, Delta_Matrix_Subset, self.omega0, Rabi, kL)
                    state = Qut.Solve_Steady_State()
                    G2_Samples.append(Emission_Class.SOCF_State_Calculation(state, Green_Far_1_SubSet, Green_Far_2_SubSet, Sp, Sm))
        else:
            if SOCF_Type[:2] == "G2":
                for sample in Samples:
                    Gamma_Matrix_Subset = Frequency_Class(Emission_Class.Generate_Subset_Matrix(sample, self.Gamma_Matrix.Nat), "Nat")
                    Delta_Matrix_Subset = Frequency_Class(Emission_Class.Generate_Subset_Matrix(sample, self.Delta_Matrix.Nat), "Nat")
                    Positions_Subset = Distance_Class(Emission_Class.Generate_Subset_of_a_List(sample, self.Positions.Nat), "Nat")
                    Qut = Qutip_Solver(Positions_Subset, Gamma_Matrix_Subset, Delta_Matrix_Subset, self.omega0, Rabi, kL)
                    state = Qut.Calculate_State_at_t(t)
                    G2_Samples.append(Emission_Class.SOCF_State_Calculation(state, Gamma_Matrix_Subset.Nat, Gamma_Matrix_Subset.Nat, Sp, Sm))
            elif SOCF_Type[:2] == "g2":
                for sample in Samples:
                    Gamma_Matrix_Subset = Frequency_Class(Emission_Class.Generate_Subset_Matrix(sample, self.Gamma_Matrix.Nat), "Nat")
                    Delta_Matrix_Subset = Frequency_Class(Emission_Class.Generate_Subset_Matrix(sample, self.Delta_Matrix.Nat), "Nat")
                    Green_Far_1_SubSet = Emission_Class.Generate_Subset_Matrix(sample, self.Far_1_Matrix)
                    Green_Far_2_SubSet = Emission_Class.Generate_Subset_Matrix(sample, self.Far_2_Matrix)
                    Qut = Qutip_Solver(Positions_Subset, Gamma_Matrix_Subset, Delta_Matrix_Subset, self.omega0, Rabi, kL)
                    state = Qut.Calculate_State_at_t(t)
                    G2_Samples.append(Emission_Class.SOCF_State_Calculation(state, Green_Far_1_SubSet, Green_Far_2_SubSet, Sp, Sm))

        return np.mean(G2_Samples)

    def Nearest_Sampling(self, SOCF_Type:str, Sample_Size:int, t, Overlap:bool, Rabi:Frequency_Class=None, kL:Inv_Metre_Class=None):
        Samples = Emission_Class.Generate_Subset_Nearest_Neighbour(self.Positions, Sample_Size, Overlap)
        Sp = Qutip_Solver.Generate_m_Operators(len(self.Positions.Nat), qutip.sigmap())
        Sm = Qutip_Solver.Generate_m_Operators(len(self.Positions.Nat), qutip.sigmam())
        G2_Samples = []
        if t == 0:
            if SOCF_Type[:2] == "G2":
                for sample in Samples:
                    Gamma_Matrix_Subset = Emission_Class.Generate_Subset_Matrix(sample, self.Gamma_Matrix.Nat)
                    G2_Samples.append(Emission_Class.Garcia_Calculation_Alt(Gamma_Matrix_Subset))
            elif SOCF_Type[:2] == "g2":
                if self.Same_Far == False:
                    for sample in Samples:
                        Green_Far_1_SubSet = Emission_Class.Generate_Subset_Matrix(sample, self.Far_1_Matrix)
                        Green_Far_2_SubSet = Emission_Class.Generate_Subset_Matrix(sample, self.Far_2_Matrix)
                        G2_Samples.append(Emission_Class.SOCF_Inverted_Calculation(Green_Far_1_SubSet, Green_Far_2_SubSet))
                else:
                    for sample in Samples:
                        Green_Far_1_SubSet = Emission_Class.Generate_Subset_Matrix(sample, self.Far_1_Matrix)
                        G2_Samples.append(Emission_Class.Garcia_Calculation_Alt(Green_Far_1_SubSet))
        if t == "inf":
            if SOCF_Type[:2] == "G2":
                for sample in Samples:
                    Gamma_Matrix_Subset = Frequency_Class(Emission_Class.Generate_Subset_Matrix(sample, self.Gamma_Matrix.Nat), "Nat")
                    Delta_Matrix_Subset = Frequency_Class(Emission_Class.Generate_Subset_Matrix(sample, self.Delta_Matrix.Nat), "Nat")
                    Positions_Subset = Distance_Class(Emission_Class.Generate_Subset_of_a_List(sample, self.Positions.Nat), "Nat")
                    Qut = Qutip_Solver(Positions_Subset, Gamma_Matrix_Subset, Delta_Matrix_Subset, self.omega0, Rabi, kL)
                    state = Qut.Solve_Steady_State()
                    G2_Samples.append(Emission_Class.SOCF_State_Calculation(state, Gamma_Matrix_Subset.Nat, Gamma_Matrix_Subset.Nat, Sp, Sm))
            elif SOCF_Type[:2] == "g2":
                for sample in Samples:
                    Gamma_Matrix_Subset = Frequency_Class(Emission_Class.Generate_Subset_Matrix(sample, self.Gamma_Matrix.Nat), "Nat")
                    Delta_Matrix_Subset = Frequency_Class(Emission_Class.Generate_Subset_Matrix(sample, self.Delta_Matrix.Nat), "Nat")
                    Green_Far_1_SubSet = Emission_Class.Generate_Subset_Matrix(sample, self.Far_1_Matrix)
                    Green_Far_2_SubSet = Emission_Class.Generate_Subset_Matrix(sample, self.Far_2_Matrix)
                    Qut = Qutip_Solver(Positions_Subset, Gamma_Matrix_Subset, Delta_Matrix_Subset, self.omega0, Rabi, kL)
                    state = Qut.Solve_Steady_State()
                    G2_Samples.append(Emission_Class.SOCF_State_Calculation(state, Green_Far_1_SubSet, Green_Far_2_SubSet, Sp, Sm))
        else:
            if SOCF_Type[:2] == "G2":
                for sample in Samples:
                    Gamma_Matrix_Subset = Frequency_Class(Emission_Class.Generate_Subset_Matrix(sample, self.Gamma_Matrix.Nat), "Nat")
                    Delta_Matrix_Subset = Frequency_Class(Emission_Class.Generate_Subset_Matrix(sample, self.Delta_Matrix.Nat), "Nat")
                    Positions_Subset = Distance_Class(Emission_Class.Generate_Subset_of_a_List(sample, self.Positions.Nat), "Nat")
                    Qut = Qutip_Solver(Positions_Subset, Gamma_Matrix_Subset, Delta_Matrix_Subset, self.omega0, Rabi, kL)
                    state = Qut.Calculate_State_at_t(t)
                    G2_Samples.append(Emission_Class.SOCF_State_Calculation(state, Gamma_Matrix_Subset.Nat, Gamma_Matrix_Subset.Nat, Sp, Sm))
            elif SOCF_Type[:2] == "g2":
                for sample in Samples:
                    Gamma_Matrix_Subset = Frequency_Class(Emission_Class.Generate_Subset_Matrix(sample, self.Gamma_Matrix.Nat), "Nat")
                    Delta_Matrix_Subset = Frequency_Class(Emission_Class.Generate_Subset_Matrix(sample, self.Delta_Matrix.Nat), "Nat")
                    Green_Far_1_SubSet = Emission_Class.Generate_Subset_Matrix(sample, self.Far_1_Matrix)
                    Green_Far_2_SubSet = Emission_Class.Generate_Subset_Matrix(sample, self.Far_2_Matrix)
                    Qut = Qutip_Solver(Positions_Subset, Gamma_Matrix_Subset, Delta_Matrix_Subset, self.omega0, Rabi, kL)
                    state = Qut.Calculate_State_at_t(t)
                    G2_Samples.append(Emission_Class.SOCF_State_Calculation(state, Green_Far_1_SubSet, Green_Far_2_SubSet, Sp, Sm))

        return np.mean(G2_Samples)
   
    def Gardiner_Sampling(self, SOCF_Type:str, Num_of_Samples:int, t, Rabi:Frequency_Class=None, kL:Inv_Metre_Class=None):
        Samples = Emission_Class.Generate_Subset_List(len(self.Positions), Num_of_Samples, 2)

        G2_Samples = []
        if t == 0:
            state = Qutip_Solver.Fully_Excited_State(2)
            if SOCF_Type[:2] == "G2":
                for sample in Samples:
                    Gamma_Matrix_Subset = Emission_Class.Generate_Subset_Matrix(sample, self.Gamma_Matrix.Nat)
                    G2_Samples.append(Emission_Class.Gardiner_Approximation(state, Gamma_Matrix_Subset, Gamma_Matrix_Subset))
            elif SOCF_Type[:2] == "g2":
                if self.Same_Far == False:
                    for sample in Samples:
                        Green_Far_1_SubSet = Emission_Class.Generate_Subset_Matrix(sample, self.Far_1_Matrix)
                        Green_Far_2_SubSet = Emission_Class.Generate_Subset_Matrix(sample, self.Far_2_Matrix)
                        G2_Samples.append(Emission_Class.Gardiner_Approximation(state, Green_Far_1_SubSet, Green_Far_2_SubSet))
                else:
                    for sample in Samples:
                        Green_Far_1_SubSet = Emission_Class.Generate_Subset_Matrix(sample, self.Far_1_Matrix)
                        G2_Samples.append(Emission_Class.Gardiner_Approximation(state, Green_Far_1_SubSet, Green_Far_1_SubSet))
        if t == "inf":
            if SOCF_Type[:2] == "G2":
                for sample in Samples:
                    Gamma_Matrix_Subset = Frequency_Class(Emission_Class.Generate_Subset_Matrix(sample, self.Gamma_Matrix.Nat), "Nat")
                    Delta_Matrix_Subset = Frequency_Class(Emission_Class.Generate_Subset_Matrix(sample, self.Delta_Matrix.Nat), "Nat")
                    Positions_Subset = Distance_Class(Emission_Class.Generate_Subset_of_a_List(sample, self.Positions.Nat), "Nat")
                    Qut = Qutip_Solver(Positions_Subset, Gamma_Matrix_Subset, Delta_Matrix_Subset, self.omega0, Rabi, kL)
                    state = Qut.Solve_Steady_State()
                    G2_Samples.append(Emission_Class.Gardiner_Approximation(state, Gamma_Matrix_Subset.Nat, Gamma_Matrix_Subset.Nat))
            elif SOCF_Type[:2] == "g2":
                for sample in Samples:
                    Gamma_Matrix_Subset = Frequency_Class(Emission_Class.Generate_Subset_Matrix(sample, self.Gamma_Matrix.Nat), "Nat")
                    Delta_Matrix_Subset = Frequency_Class(Emission_Class.Generate_Subset_Matrix(sample, self.Delta_Matrix.Nat), "Nat")
                    Green_Far_1_SubSet = Emission_Class.Generate_Subset_Matrix(sample, self.Far_1_Matrix)
                    Green_Far_2_SubSet = Emission_Class.Generate_Subset_Matrix(sample, self.Far_2_Matrix)
                    Qut = Qutip_Solver(Positions_Subset, Gamma_Matrix_Subset, Delta_Matrix_Subset, self.omega0, Rabi, kL)
                    state = Qut.Solve_Steady_State()
                    G2_Samples.append(Emission_Class.Gardiner_Approximation(state, Green_Far_1_SubSet, Green_Far_2_SubSet))
        else:
            if SOCF_Type[:2] == "G2":
                for sample in Samples:
                    Gamma_Matrix_Subset = Frequency_Class(Emission_Class.Generate_Subset_Matrix(sample, self.Gamma_Matrix.Nat), "Nat")
                    Delta_Matrix_Subset = Frequency_Class(Emission_Class.Generate_Subset_Matrix(sample, self.Delta_Matrix.Nat), "Nat")
                    Positions_Subset = Distance_Class(Emission_Class.Generate_Subset_of_a_List(sample, self.Positions.Nat), "Nat")
                    Qut = Qutip_Solver(Positions_Subset, Gamma_Matrix_Subset, Delta_Matrix_Subset, self.omega0, Rabi, kL)
                    state = Qut.Calculate_State_at_t(t)
                    G2_Samples.append(Emission_Class.Gardiner_Approximation(state, Green_Far_1_SubSet, Green_Far_2_SubSet))
            elif SOCF_Type[:2] == "g2":
                for sample in Samples:
                    Gamma_Matrix_Subset = Frequency_Class(Emission_Class.Generate_Subset_Matrix(sample, self.Gamma_Matrix.Nat), "Nat")
                    Delta_Matrix_Subset = Frequency_Class(Emission_Class.Generate_Subset_Matrix(sample, self.Delta_Matrix.Nat), "Nat")
                    Green_Far_1_SubSet = Emission_Class.Generate_Subset_Matrix(sample, self.Far_1_Matrix)
                    Green_Far_2_SubSet = Emission_Class.Generate_Subset_Matrix(sample, self.Far_2_Matrix)
                    Qut = Qutip_Solver(Positions_Subset, Gamma_Matrix_Subset, Delta_Matrix_Subset, self.omega0, Rabi, kL)
                    state = Qut.Calculate_State_at_t(t)
                    G2_Samples.append(Emission_Class.Gardiner_Approximation(state, Green_Far_1_SubSet, Green_Far_2_SubSet))

        return np.mean(G2_Samples)

    """
    def Under_Sampling_Distrbution(self, Sample_Size_Set, Num_of_Samples, a, N):
        Filename = "Gardiner_Work/Data_Files/Distribution/G2_Samples_" + str(a) + "_" + str(N) + "_"
        G2_Samples = []
        for Sample_Size in Sample_Size_Set:
            Samples = Emission_Class.Generate_Subset_List(len(self.Positions.Nat), Num_of_Samples, Sample_Size)
            i = 1
            for sample in Samples:
                print(sample)
                print(i)
                i = i+1
                Gamma_Matrix_Subset = Emission_Class.Generate_Subset_Matrix(sample, self.Gamma_Matrix.Nat)
                G2 = np.real(Emission_Class.Garcia_Homogenious_Calculation(Gamma_Matrix_Subset))
                Read_Write_Class.Write_to_File(Filename + str(Sample_Size), G2)
            G2_Samples.append(Read_Write_Class.Read_In_Data(Filename + str(Sample_Size)))
        return G2_Samples

    def Under_Sampling_Distrbution_g2(self, Sample_Size_Set, Num_of_Samples, a, N):
        Filename = "Gardiner_Work/Data_Files/Distribution/g2xy_Samples_" + str(a) + "_" + str(N) + "_"
        G2_Samples = []
        for Sample_Size in Sample_Size_Set:
            Samples = Emission_Class.Generate_Subset_List(len(self.Positions.Nat), Num_of_Samples, Sample_Size)
            i = 1
            for sample in Samples:
                print(sample)
                print(i)
                i = i+1
                A1_Subset = Emission_Class.Generate_Subset_Matrix(sample, self.Far_1_Matrix)
                A2_Subset = Emission_Class.Generate_Subset_Matrix(sample, self.Far_2_Matrix)
                G2 = np.real(Emission_Class.SOCF_Inverted_Calculation(A1_Subset, A2_Subset))
                Read_Write_Class.Write_to_File(Filename + str(Sample_Size), G2)
            G2_Samples.append(Read_Write_Class.Read_In_Data(Filename + str(Sample_Size)))
        return G2_Samples
    """

#Emission Rate Calculation
    def Exact_Emission_Rate(self, t_max, res=200):
        qut = Qutip_Solver(self.Gamma_Matrix, self.Delta_Matrix, self.omega0)
        t, Exp = qut.Emission_Rate_over_time(t_max, 200, True)
        return t, Exp

    def Underestimate_Emission_Rate(self, t_max, S, m, res=200):
        Samples = Emission_Class.Generate_Subset_List(len(self.Positions.Nat), S, m)
        Exps = np.zeros(res)
        t = []
        for sample in Samples:
            Gamma_Matrix_Subset = Frequency_Class(Emission_Class.Generate_Subset_Matrix(sample, self.Gamma_Matrix.Nat), "Nat")
            Delta_Matrix_Subset = Frequency_Class(Emission_Class.Generate_Subset_Matrix(sample, self.Delta_Matrix.Nat), "Nat")
            Positions_Subset = Distance_Class(Emission_Class.Generate_Subset_of_a_List(sample, self.Positions.Nat), "Nat")
            Qut = Qutip_Solver(Gamma_Matrix_Subset, Delta_Matrix_Subset, self.omega0)
            t, Exp = Qut.Emission_Rate_over_time(t_max, res, True)
            Exps = Exps + Exp
        
        mean_Exps = []
        for i in range(len(Exps)):
            mean_Exps.append(Exps[i]/S)
        return t, mean_Exps
