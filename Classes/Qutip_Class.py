import sys

from .Unit_Converter_Classes import *
from .SMUTHI_Environment_Class import *

import cmath
import numpy as np
from sympy import Options
from qutip import *
import time


class Qutip_Solver:
     
    def __init__(self, Gamma_Matrix:Frequency_Class, Delta_Matrix:Frequency_Class, omega_0:Frequency_Class, 
                 Positons:Distance_Class = Distance_Class([], "Nat") ,Rabi:Frequency_Class=Frequency_Class(0, "Nat"), 
                 kL:Inv_Metre_Class=Inv_Metre_Class([0,0,0], "Nat"), T:Temp_Class=Temp_Class(0, "Nat")):
        option = Options()
        option.atol = 10**-11
        option.rtol = 10**-11

        self.Positions = Positons.Nat
        self.N_Dipoles = len(Gamma_Matrix.Nat)

        self.Gamma_Matrix = Gamma_Matrix.Nat
        self.Delta_Matrix = Delta_Matrix.Nat

        self.omega_0 = omega_0.Nat
        self.Rabi = Rabi.Nat
        self.kL = kL.Nat
        self.omega_L = np.linalg.norm(kL.Nat)
        self.Temp = T.Nat

        self.SigmaPlus = Qutip_Solver.Generate_m_Operators(self.N_Dipoles, qutip.sigmap())
        self.SigmaMinus = Qutip_Solver.Generate_m_Operators(self.N_Dipoles, qutip.sigmam())

    @staticmethod
    def Generate_m_Operators(m, Op:Qobj):
        Op_Set = []
        for i in range(m):
            temp = Op
            for j in range(m):
                if i == j:
                    pass
                elif i < j:
                    temp = tensor(temp, identity(2))
                elif i > j:
                    temp = tensor(identity(2), temp)
            Op_Set.append(temp)
        return Op_Set

#State Options
    @staticmethod
    def Fully_Excited_State(m):
        tensor_prod = []
        for i in range(m):
            tensor_prod.append(fock(2,0))
        return tensor(tensor_prod)

    def Fully_Ground_State(self):
        tensor_prod = []
        for i in range(self.N_Dipoles):
            tensor_prod.append(fock(2,1))
        return tensor(tensor_prod)
    
    def N(self):
        N = 0
        if self.Temp == 0:
            N=0
        else:
            N = 1/(np.exp(self.omega_0/self.Temp) - 1)
        return N


#Master Equation
    def Hamiltonian(self):
        VonNeumann = 0 
        for i in range(self.N_Dipoles):
            VonNeumann = VonNeumann + (self.omega_0 - self.omega_L) * self.SigmaPlus[i] * self.SigmaMinus[i] 
        
        Driving = 0
        if self.Rabi != 0:
            for i in range(self.N_Dipoles):
                exp_1 = cmath.exp(-1j * np.dot(self.kL, self.Positions[i]))
                Driving = Driving - 0.5 * self.Rabi * (exp_1 * self.SigmaPlus[i] + np.conj(exp_1) * self.SigmaMinus[i])
        
        Coupling = 0
        if self.N_Dipoles != 1:
            for i in range(self.N_Dipoles):
                for j in range(self.N_Dipoles):
                    if i != j:
                        Coupling = Coupling + self.Delta_Matrix[i][j] * self.SigmaPlus[i] * self.SigmaMinus[j]
        return VonNeumann + Driving + Coupling
   
    def Collapse_Operator_List(self):
        Collapse_operators = []
        Gamma_Eigen = np.linalg.eigh(self.Gamma_Matrix)
        Gamma_Eigenvals = Gamma_Eigen[0]
        Gamma_Eigenvecs = Gamma_Eigen[1]
        N = self.N()
        for i in range(self.N_Dipoles):
            temp_collapse_i = 0
            for j in range(self.N_Dipoles):
                temp_collapse_i = temp_collapse_i + cmath.sqrt((N+1)*Gamma_Eigenvals[i]) * Gamma_Eigenvecs[j][i] * self.SigmaMinus[j]
            Collapse_operators.append(temp_collapse_i)
        if self.Temp != 0:
            for i in range(self.N_Dipoles):
                temp_collapse_i = 0
                for j in range(self.N_Dipoles):
                    temp_collapse_i = temp_collapse_i + cmath.sqrt(N*Gamma_Eigenvals[i]) * Gamma_Eigenvecs[j][i] * self.SigmaPlus[j]
                Collapse_operators.append(temp_collapse_i)

        return Collapse_operators, Gamma_Eigenvals

    def Solve_Steady_State(self):
        C_ops, _ = self.Collapse_Operator_List() 
        rho_ss = steadystate(self.Hamiltonian(), C_ops)
        return rho_ss




    def Calculate_State_at_t(self, t):
        psi0 = self.Fully_Excited_State(self.N_Dipoles)
        H = self.Hamiltonian()
        Collapse_Operators, Gamma_Evals = self.Collapse_Operator_List()
        times = np.linspace(0.0, t, 100)
        N_ops = []
        result = mesolve(H, psi0, times, Collapse_Operators, N_ops, options=Options(nsteps=1000000)).states
        return result[len(result)-1]

    def Analytical_Single_SS(self):
        omega = self.omega_0 - self.omega_L
        Rabi = self.Rabi * cmath.exp(-1j * np.dot(self.kL, self.Positions[0]))
        A = 0.5*(self.Gamma_Matrix[0][0]**2) + 2*(omega**2) + Rabi * np.conj(Rabi)
        r00 = (0.5* Rabi * np.conj(Rabi))/A
        r01 = (2*omega + 1j*self.Gamma_Matrix[0][0])*Rabi/(2*A)
        r10 = np.conj(r01)
        r11 = 0.5 + (0.25*(self.Gamma_Matrix[0][0]**2) + omega**2 )/A
        return [[r00, r01],[r10, r11]]

    def Numerical_Steady_State(self, n, d, Threshold):
        psi0 = self.Fully_Excited_State(self.N_Dipoles)
        H = self.Hamiltonian()
        Collapse_Operators, Gamma_Evals = self.Collapse_Operator_List()
        Gamma_0 =  Frequency_Class(((self.omega_0**3)/(3*np.pi)) * np.dot(np.conj(d), d), "Nat").Nat
        Louivillian = liouvillian(H, Collapse_Operators).full()
        Threshold_met = False
        tn = n/Gamma_0
        times = np.linspace(0, tn, 100)
        N_ops = []
        counter = 0
        while Threshold_met == False:
            #print(psi0)
            counter = counter+1
            result = mesolve(H, psi0, times, Collapse_Operators, N_ops, options=Options(nsteps=100000)).states
            rho = result[len(result)-1]
            psi0 = rho.copy()
            rate_of_Change = np.matmul(Louivillian, np.array(rho.full()).flatten("F"))/self.N_Dipoles**2
            #print(np.linalg.norm(rate_of_Change))
            if np.linalg.norm(rate_of_Change) <= Threshold:
                Threshold_met = True
                #print("Steady Found at t=" + str(counter * n))
        return psi0

#Evolution
    def Calculate_Dynamics(self, t:float, Param_Scale:float, resolution:int, psi0=None):
        if psi0==None:
            psi0 = Qutip_Solver.Fully_Excited_State(self.N_Dipoles)
        H = self.Hamiltonian()
        Collapse_Operators, Gamma_Evals = self.Collapse_Operator_List()
        times = np.linspace(0.0, t/Param_Scale, resolution)
        N_ops = []
        result = mesolve(H, psi0, times, Collapse_Operators, N_ops, options=Options(nsteps=100000)).states
        return times*Param_Scale, result

    def Emission_Rate_over_time(self, t_max, resolution, normalize:bool, psi0=None):
        if psi0==None:
            psi0 = Qutip_Solver.Fully_Excited_State(self.N_Dipoles)

        Gamma = self.Gamma_Matrix[0][0]
        t = np.linspace(0, t_max/Gamma, resolution)
        Collapse_Ops, Gamma_Eigenvals = self.Collapse_Operator_List()
        Gamma_0 = np.mean(Gamma_Eigenvals)
        Ops = []
        for i in range(self.N_Dipoles):
            Ops.append(Collapse_Ops[i].dag()*Collapse_Ops[i])
            
        result = mesolve(self.Hamiltonian(), psi0, t, c_ops=Collapse_Ops, e_ops=Ops, options=Options(nsteps=100000))
        total_Emission_Rate = []
        for i in range(resolution):
            temp_total = 0
            for j in range(self.N_Dipoles):
                temp_total = temp_total + result.expect[j][i]
            if normalize == True:
                total_Emission_Rate.append(temp_total/(self.N_Dipoles*Gamma_0))
            else:
                total_Emission_Rate.append(temp_total)
        
        total_Emission_Rate = total_Emission_Rate
        return t*Gamma, total_Emission_Rate

    def Quantum_Regression_Run(self, t, Param_Scale, resolution, State, Matrix):
        times = np.linspace(0, t/Param_Scale, resolution)
        Hamiltonian = self.Hamiltonian()
        Collapse_Operators, Gamma_Eigenvals = self.Collapse_Operator_List()
        
        G1_Operators = []
        Green_Far_G1 = []
        for i in range(self.N_Dipoles):
            for j in range(self.N_Dipoles):
                G1_Operators.append(self.SigmaPlus[i]*self.SigmaMinus[j])
                Green_Far_G1.append(Matrix[j][i])

        G1_t = 0 #Constant value as t is fixed, defined by the State t0
        for i in range(len(G1_Operators)):
            G1_t = G1_t + Green_Far_G1[i] * expect(G1_Operators[i], State)
        print("G1(t) done")

        result = mesolve(Hamiltonian, State, times, Collapse_Operators, G1_Operators, options=Options(nsteps=100000))
        G1_tau_Array = []
        for i in range(resolution):
            G1_tau = 0
            for j in range(len(G1_Operators)):
                G1_tau = G1_tau + Green_Far_G1[j] * result.expect[j][i]
            G1_tau_Array.append(G1_tau)
        print("G1(t+tau) done")

        G2_tau_Array = np.zeros(resolution)
        for i in range(self.N_Dipoles):
            for j in range(self.N_Dipoles):
                for k in range(self.N_Dipoles):
                    for l in range(self.N_Dipoles):
                        Chi = self.SigmaMinus[l]*State*self.SigmaPlus[i]
                        Operator = self.SigmaPlus[j]*self.SigmaMinus[k]
                        result = mesolve(Hamiltonian, Chi, times, Collapse_Operators, [Operator], options=Options(nsteps=100000))
                        G2_tau_Array = G2_tau_Array +  Matrix[l][i] * Matrix[k][j] * result.expect[0]
        print("G2(t, t+tau) done")

        g2 = []
        for i in range(resolution):
            g2.append(G2_tau_Array[i]/(G1_tau_Array[i] * G1_t))
        return times*Param_Scale, g2

    def Find_Max_Emission(self, state, t_max, normalize:bool):
        tGamma, Emission = self.Emission_Rate_over_time(t_max, state, 5000, normalize)
        max_index = Emission.index(max(Emission))
        return tGamma[max_index], max(Emission)


#Correlation Functions
    def Generate_t_Expectation_Matrix(self, state):
        G1_Mat = np.zeros([self.N_Dipoles, self.N_Dipoles], dtype="complex_")
        G2_Mat = np.zeros([self.N_Dipoles, self.N_Dipoles, self.N_Dipoles, self.N_Dipoles], dtype="complex_")
        for i in range(self.N_Dipoles):
            for j in range(self.N_Dipoles):
                spsm = self.SigmaPlus[i] * self.SigmaMinus[j]
                G1_Mat[i][j]  = expect(spsm, state)
        for i in range(self.N_Dipoles):
            for j in range(self.N_Dipoles):
                for k in range(self.N_Dipoles):
                    for l in range(self.N_Dipoles):
                        spsm = self.SigmaPlus[i] * self.SigmaPlus[j] * self.SigmaMinus[k] * self.SigmaMinus[l]
                        G2_Mat[i][j][k][l] = expect(spsm, state)
        return G1_Mat, G2_Mat
