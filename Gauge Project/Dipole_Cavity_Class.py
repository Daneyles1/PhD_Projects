import numpy as np
from scipy.linalg import expm #for matrix exponential
from matplotlib import pyplot as plt #for plotting
import scipy.sparse.linalg as spla
from scipy.sparse import  kron #for kron product
import qutip #for operators
import copy #to generate a copy of an array 
from scipy.integrate import solve_ivp #for ODE solver


class Dipole_Cavity_System: #Dipole cavity class

    H1_Color = 'pink' #Colors for the Plots
    H12_Color = 'green'
    PHP_Color = "black"
    h10_Color = "cyan"

    def __init__(self, mmult, nmult, Ed, beta, detuning, Alternate_Basis=False, x_min=-10, x_max=10, n_points=200, matter_modes=20):
        #Object parameters
        self.mmult = mmult #Number of matter levels
        self.nmult = nmult #Number of cavity levels
        self.Ed = Ed
        self.beta = beta
        self.detuning = detuning
        self.xi_min = x_min
        self.xi_max = x_max
        self.n_points = 200
        self.m = matter_modes
        self.Alternate_Basis = Alternate_Basis

        #Discretize the spatial domain using finite differences
        self.xi = np.linspace(self.xi_min, self.xi_max, n_points)  
        self.dx = self.xi[1] - self.xi[0]

        #Create the kinetic energy operator (second derivative) and Diagonalise the potential: Generate matter Hamitlonian
        T = -0.5*self.Ed*(np.diag(-2*np.ones(self.n_points)) + np.diag(np.ones(self.n_points-1),1) + np.diag(np.ones(self.n_points- 1),-1))/self.dx**2
        V = np.diag(self.V()) 
        Hm = T + V

        #Compute eigenvalues and eigenvectors of the Hamiltonian
        self.eigenvalues, eigenfunctions = spla.eigsh(Hm, k=self.m, which='SM')  # k=5 for 5 lowest eigenvalues
        eigenfunctions = np.transpose(eigenfunctions)

        #Compute zeta matrix 
        ZetaMat1 = self.compute_zeta1_matrix(eigenfunctions)
        self.zeta_matp = (np.conjugate(ZetaMat1).transpose() + ZetaMat1)

        #Compute transition matrix
        self.omega_trans = np.zeros([self.m, self.m])
        for i in range(self.m):
            for j in range(self.m):
                self.omega_trans[i][j] = self.eigenvalues[i] - self.eigenvalues[j]
        
        self.omega_0 = abs(self.omega_trans[1][0])
        self.Omega = self.detuning * self.omega_0

        #Compute the correction so at eta=0 the GS energy is 0
        self.Correction = abs(min(np.linalg.eigvalsh(self.Hmatter(self.mmult) + self.Hcavity(self.mmult))))

    #Potential of dipole Hamiltonian, double well
    def V(self): 
        return -(self.Ed * self.beta) / 2 * self.xi**2 + (self.Ed / 4) * self.xi**4

    #Compute integral using the trapezoid rule
    def compute_integral(self, k, l, eigenfunctions): 
        if (k % 2 == 0 and l % 2 == 0) or (k % 2 != 0 and l % 2 != 0) or k < l:
            return 0  # No calculation needed
        else:
            # Compute the integral if the condition is not met
            integrand = np.conj(eigenfunctions[k]) * self.xi * eigenfunctions[l]
            integral = np.sum(integrand) * self.dx
            return integral

    #Compute zeta1 matrix
    def compute_zeta1_matrix(self, eigenfunctions):
        ZetaMat1 = np.zeros((self.m, self.m), dtype=complex)  # Initialize the matrix
        
        for k in range(self.m):
            for l in range(self.m):  # Calculate for all pairs of k and l
                # Compute the integral for the matrix element
                ZetaMat1[k, l] = self.compute_integral(k, l, eigenfunctions)
        
        return np.round(ZetaMat1,15)

    #Operator to transform to perturbation basis 
    def New_Basis_Operator(self, mmult): 
        #This code only currently does superposition of |e,0> and |g,1> but could be extended to all degen. states
        Old_Basis = [] #Generate the basis of the system
        for i in range(mmult):
            Matter_Vec = np.zeros(mmult)
            Matter_Vec[i] = 1
            for j in range(self.nmult):
                Cavity_Vec = np.zeros(self.nmult)
                Cavity_Vec[j] = 1
                Vec = np.kron(Matter_Vec, Cavity_Vec)
                Old_Basis.append(Vec)

        Old_Basis = Old_Basis  # Ensure proper structure
        New_Basis = copy.deepcopy(Old_Basis) #Create a copy of the basis in memory

        g1 = Old_Basis[1] 
        e0 = Old_Basis[self.nmult]
        psi_Plus = (1/np.sqrt(2)) * (e0 + 1j * g1) #generate superposition
        psi_Minus = (1/np.sqrt(2)) * (e0 - 1j * g1)
        New_Basis[self.nmult] = psi_Minus #assign to new basis 
        New_Basis[1] = psi_Plus

        # Ensure all basis vectors are 1D
        E = np.column_stack(Old_Basis)  
        F = np.column_stack(New_Basis)  

        # Compute transformation matrix
        U = F.T.conj() @ E

        return np.array(U)

    #Create Annihilation operator
    def aminus(self): 
        return qutip.destroy(self.nmult).full() #Qutip a operator

    #Create Creation operator
    def aplus(self): 
        return qutip.create(self.nmult).full() #Qutip a operator (.full()) just makes it a normal list rather than a qutip object)

    #Create Matter Hamiltonian 
    def Hmatter(self, mmult): 
        return kron(np.diag(self.eigenvalues[:mmult]), np.eye(self.nmult))

    # Create Cavity Hamiltonian
    def Hcavity(self, mmult): 
        Nop = kron(np.eye(mmult),self.aplus() @ self.aminus())
        return self.Omega * (Nop + np.eye(mmult * self.nmult) / 2)

    #Create Polarisation hamiltonian
    def Hpol(self, mmult, etaD, alpha): 
        return (alpha**2 * etaD**2 * self.Omega / self.zeta_matp[0, 1]**2) * kron(self.zeta_matp[:mmult, :mmult] @ self.zeta_matp[:mmult, :mmult], np.eye(self.nmult))

    #Create Interaction Hamiltonian
    def Hinteraction(self, mmult, etaD, alpha): 
        multi_term = (-1j * self.Omega * alpha * (etaD / self.zeta_matp[0, 1]) * kron(self.zeta_matp[:mmult, :mmult], self.aplus() - self.aminus())) 
        coul_term = 1j * (1 - alpha) * (etaD / self.zeta_matp[0, 1]) * kron(self.omega_trans[:mmult, :mmult] * self.zeta_matp[:mmult, :mmult], self.aplus() + self.aminus()) 
        A_Squared = (etaD**2 * self.Ed / (2 * self.zeta_matp[0, 1]**2)) * (1 - alpha)**2 * kron(np.eye(mmult), (self.aplus() + self.aminus()) @ (self.aplus() + self.aminus()))
        return multi_term + coul_term + A_Squared

    #Create Projection operator
    def Proj(self,mmult): 
        return kron(np.diag([1 if i in [0, 1] else 0 for i in range(mmult)]), np.eye(self.nmult))

    #Create PWZ gauge operator
    def Ualpha(self, etaD, alpha, alpha2): 
        Matrix = 1j * (alpha - alpha2) * (etaD / self.zeta_matp[0, 1]) * kron(self.zeta_matp[:self.mmult, :self.mmult], self.aplus() + self.aminus())
        return expm(Matrix.toarray())

    #Savasta TL 'gauge' operator
    def Talpha(self, etaD, alpha, alpha2): 
        Matrix = 1j * (alpha - alpha2) * (etaD / self.zeta_matp[0, 1]) * kron(self.zeta_matp[:2, :2], self.aplus() + self.aminus())
        return expm(Matrix.toarray())

# ---------------------------------------------
# Hamiltonian Generators
# ---------------------------------------------

    #Generate Full Hamiltonian
    def H(self, mmult, etaD, alpha): 
        Halpha = self.Hmatter(mmult) + self.Hcavity(mmult) + self.Hinteraction(mmult, etaD, alpha) + self.Hpol(mmult, etaD, alpha) 
        Halpha_Corr = Halpha + self.Correction * np.eye(self.nmult*mmult)
        if self.Alternate_Basis == False:
            return Halpha_Corr
        else:
            U = self.New_Basis_Operator(mmult) #Transform to new basis
            return U @ Halpha_Corr @ np.conj(np.transpose(U))

    #generate Quantum Rabi model Hamiltonian
    def H2(self, etaD, alpha): 
        return self.H(2, etaD, alpha)

    #Generate Numerical truncated hamiltonian
    def PHP(self, etaD, alpha): 
        return self.H(self.mmult, etaD, alpha)[:2*self.nmult, :2*self.nmult]

    #Generate Savasta model
    def h10(self, eta): 
        T = self.Talpha(eta, 0, 1)
        H12 = self.H2(eta, 1)
        if self.Alternate_Basis == False:
            return T @ H12 @ np.transpose(np.conj(T))
        else:
            U_basis = self.New_Basis_Operator(2)  # Transform to pert. basis   
            Tprime = (U_basis @ T @ np.conj(np.transpose(U_basis)))
            return Tprime @ H12 @ np.transpose(np.conj(Tprime))

# ---------------------------------------------
# Observable Operators
# ---------------------------------------------

    #Coulomb gauge photon number operator in gauge alpha
    def CG_Photon_Number(self, mmult, alpha, eta): 
        Ua = self.Ualpha(eta, alpha, 0)
        N = kron(np.eye(self.mmult), self.aplus() @ self.aminus())

        Nop = ( Ua @ N @ np.conj(np.transpose(Ua)) )[:mmult*self.nmult, :mmult*self.nmult]
        if self.Alternate_Basis == False:
            return Nop
        else: 
            U_basis = self.New_Basis_Operator(mmult)            
            return (U_basis @ Nop @ np.conj(np.transpose(U_basis)))

    #Coulomb gauge matter excitation operator in gauge alpha
    def CG_Matter_Exciations(self, mmult, alpha, eta): 
        Ua = self.Ualpha(eta, alpha, 0)

        Matter_Vec = np.zeros([self.mmult,self.mmult])
        Cavity_Vec = np.zeros([self.nmult, self.nmult])
        Matter_Vec[0][0] = 1
        GS = np.kron(Matter_Vec, np.eye(self.nmult))
        Excitation = np.eye(self.mmult*self.nmult)- GS
        Oop = ( Ua @ Excitation @ np.conj(np.transpose(Ua)) )[:mmult*self.nmult, :mmult*self.nmult]
        if self.Alternate_Basis == False:
            return Oop
        else: 
            U_basis = self.New_Basis_Operator(mmult)            
            return (U_basis @ Oop @ np.conj(np.transpose(U_basis)))

    #Coulomb gauge field momentum operator in gauge alpha
    def CG_Field_Momentum(self, mmult, alpha, eta): 
        if alpha != 0:
            Ua = self.Ualpha(eta, alpha, 0)
            Pi = 1j * np.kron(np.eye(self.mmult), self.aplus() - self.aminus())
            Piop = ( Ua @ Pi @ np.conj(np.transpose(Ua)) )[:mmult*self.nmult, :mmult*self.nmult]
        else:
            Piop = 1j * np.kron(np.eye(mmult), self.aplus() - self.aminus())

        if self.Alternate_Basis == False:
            return Piop
        else: 
            U_basis = self.New_Basis_Operator(mmult)            
            return (U_basis @ Piop @ np.conj(np.transpose(U_basis)))

# ---------------------------------------------
# Plotting Functions
# ---------------------------------------------

    #Plot energy level n_levels is the number of levels included
    def Paper_Plot_Energy_Levels(self, n_Levels, Res = 20, fig_x = 8, fig_y = 6): 
        eta_Set = np.linspace(0,1.5, Res)
        H12_Set = []
        H1_Set = []
        PHP_Set = []
        for eta in eta_Set:
            H1 = self.H(self.mmult, eta, 1)
            H1_Set.append(np.linalg.eigvalsh(H1)[0:n_Levels])
            H12 = self.H2(eta, 1)
            H12_Set.append(np.linalg.eigvalsh(H12)[0:n_Levels])
            PHP = self.PHP(eta, 1)
            PHP_Set.append(np.linalg.eigvalsh(PHP)[0:n_Levels])

        plt.figure(figsize=(fig_x, fig_y))
        plt.plot(eta_Set, np.transpose(H1_Set)[0]/self.omega_0, label="$H_1$", c=self.H1_Color)
        plt.plot(eta_Set, np.transpose(H12_Set)[0]/self.omega_0, label="$H_1^{(2)}$", c=self.H12_Color)
        plt.plot(eta_Set, np.transpose(PHP_Set)[0]/self.omega_0, label="$P H_1 P$", c=self.PHP_Color)

        for i in range(n_Levels-1):
            plt.plot(eta_Set, np.transpose(H1_Set)[i+1]/self.omega_0, c=self.H1_Color)
            plt.plot(eta_Set, np.transpose(H12_Set)[i+1]/self.omega_0, c=self.H12_Color)
            plt.plot(eta_Set, np.transpose(PHP_Set)[i+1]/self.omega_0, c=self.PHP_Color)

        plt.xlabel("$\eta$")
        plt.ylabel("$E/\omega$")
        plt.ylim(bottom=0)
        plt.xlim(0,1.5)
        plt.legend()
        plt.show()

    #Plot energy transitions
    def Paper_Plot_Energy_Transition(self, n_Levels, Res = 20, fig_x = 8, fig_y = 6): 
        eta_Set = np.linspace(0,1.5, Res)
        H12_Set = []
        H1_Set = []
        PHP_Set = []
        for eta in eta_Set:
            H1 = self.H(self.mmult, eta, 1)
            H1_Set.append(np.linalg.eigvalsh(H1)[0:n_Levels])
            H12 = self.H2(eta, 1)
            H12_Set.append(np.linalg.eigvalsh(H12)[0:n_Levels])
            PHP = self.PHP(eta, 1)
            PHP_Set.append(np.linalg.eigvalsh(H12)[0:n_Levels])

        plt.figure(figsize=(fig_x, fig_y))
        plt.plot(eta_Set, np.transpose(H1_Set)[1]/self.omega_0 - np.transpose(H1_Set)[0]/self.omega_0, label="$H_1$", c=self.H1_Color)
        plt.plot(eta_Set, np.transpose(H12_Set)[1]/self.omega_0 - np.transpose(H12_Set)[0]/self.omega_0, label="$H_1^{(2)}$", c=self.H12_Color)
        plt.plot(eta_Set, np.transpose(PHP_Set)[1]/self.omega_0 - np.transpose(PHP_Set)[0]/self.omega_0, label="$H_1^{(2)}$", c=self.PHP_Color)

        for i in range(n_Levels-2):
            plt.plot(eta_Set, np.transpose(H1_Set)[i+2]/self.omega_0 - np.transpose(H1_Set)[0]/self.omega_0, c=self.H1_Color)
            plt.plot(eta_Set, np.transpose(H12_Set)[i+2]/self.omega_0 - np.transpose(H12_Set)[0]/self.omega_0, c=self.H12_Color)
            plt.plot(eta_Set, np.transpose(PHP_Set)[i+2]/self.omega_0 - np.transpose(PHP_Set)[0]/self.omega_0, c=self.PHP_Color)

        plt.xlabel("$\eta$")
        plt.ylabel("$(E_i - E_0)/\omega$")
        plt.ylim(bottom=0)
        plt.xlim(0,1.5)
        plt.legend()
        plt.show()

    #Plot some operator definition Op
    def Plot_Operator_Expectation(self, n_Levels, Op, Res, ylabel="", fig_x=8, fig_y=6): 
        eta_set = np.linspace(0, 1.5, Res)
        deta = eta_set[1] - eta_set[0]
        
        H12_Set = []
        H1_Set = []
        PHP_Set = []
        h10_Set = []

        for eta in eta_set:

            H1_temp = []
            H1 = np.round(self.H(self.mmult, eta, 1),10)
            H1_Val, H1_Vec = np.linalg.eigh(np.array(H1)) #eigenvectors come out in columns so should be transposed first
            H1_Vec = H1_Vec.transpose()
            pn = Op(self.mmult, 1, eta) #Op must be defined as def Op(self, alpha, eta): ...
            for i in range(n_Levels):
                H1_temp.append(np.dot(np.conj(H1_Vec[i]), np.dot(pn, H1_Vec[i])))
            H1_Set.append(H1_temp)

            H12_temp = []
            H12 = np.round(self.H2(eta, 1),10)
            H12_Val, H12_Vec = np.linalg.eigh(np.array(H12))
            H12_Vec = H12_Vec.transpose()
            pn2 = Op(2, 1, eta)
            for i in range(n_Levels):
                H12_temp.append(np.dot(np.conj(H12_Vec[i]), np.dot(pn2, H12_Vec[i])))
            H12_Set.append(H12_temp)

            PHP_temp = []
            PHP = np.round(self.PHP(eta, 1),10)
            PHP_Val, PHP_Vec = np.linalg.eigh(np.array(PHP))
            PHP_Vec = PHP_Vec.transpose()
            for i in range(n_Levels):
                PHP_temp.append(np.dot(np.conj(PHP_Vec[i]), np.dot(pn2, PHP_Vec[i])))
            PHP_Set.append(PHP_temp)
            
            h10_temp = []
            h10 = np.round(self.h10(eta),10)
            h10_Val, h10_Vec = np.linalg.eigh(np.array(h10))
            h10_Vec = h10_Vec.transpose()
            pn3 = Op(2, 0, eta)
            for i in range(n_Levels):
                h10_temp.append(np.dot(np.conj(h10_Vec[i]), np.dot(pn3, h10_Vec[i])))
            h10_Set.append(h10_temp)

        plt.figure(figsize=(fig_x, fig_y))
        plt.plot(eta_set, np.transpose(H1_Set)[0], c=self.H1_Color, label="$H_1$")
        plt.plot(eta_set, np.transpose(H12_Set)[0], c=self.H12_Color, label="$H_1^{(2)}$")
        plt.plot(eta_set, np.transpose(PHP_Set)[0], c=self.PHP_Color, label="$P H_1 P$")
        plt.plot(eta_set, np.transpose(h10_Set)[0], c=self.h10_Color, label="$h_1(0)$")
        if n_Levels > 1:
            for i in range(n_Levels-1):
                plt.plot(eta_set, np.transpose(H1_Set)[i+1],c=self.H1_Color)
                plt.plot(eta_set, np.transpose(H12_Set)[i+1],c=self.H12_Color)
                plt.plot(eta_set, np.transpose(PHP_Set)[i+1], c=self.PHP_Color)
                plt.plot(eta_set, np.transpose(h10_Set)[i+1], c=self.h10_Color)

        plt.legend()
        plt.xlabel("$\eta$")
        plt.ylabel(ylabel)
        plt.show()

    #Plot multipolar fidelity
    def Paper_Plot_Multipolar_Fidelity(self, Res=20):
        eta_set = np.linspace(0, 1.5, Res)
        deta = eta_set[1] - eta_set[0]

        Bound_Set = []
        H12_Set = []
        PHP_Set = []

        for eta in eta_set:

            H1 = self.H(self.mmult, eta, 1)
            H1_Val, H1_Vec = np.linalg.eigh(H1)
            H1_Vec = H1_Vec.transpose()
            H1_trunc = (H1_Vec[0].T)[:2*self.nmult]
            dot = np.abs(np.linalg.norm(H1_trunc))
            Bound_Set.append(dot**2)

            H12 = self.H2(eta, 1)
            H12_Val, H12_Vec = np.linalg.eigh(H12)
            H12_Vec = H12_Vec.transpose()
            dot2 = np.array(np.abs(np.dot(H12_Vec[0].conj(), (H1_Vec[0].T)[:2*self.nmult])))
            H12_Set.append(np.abs(dot2[0][0])**2)

            PHP = self.PHP(eta, 1)
            PHP_Val, PHP_Vec = np.linalg.eigh(PHP)
            PHP_Vec = PHP_Vec.transpose()
            dot3 = np.array(np.abs(np.dot(PHP_Vec[0].conj(), (H1_Vec[0].T)[:2*self.nmult])))
            PHP_Set.append(np.abs(dot3[0][0])**2)


        plt.plot(eta_set, Bound_Set, label="Bound", c=self.H1_Color)
        plt.plot(eta_set, H12_Set, label="$H_1^{(2)}$", c=self.H12_Color)
        plt.plot(eta_set, PHP_Set, label="$P H_1 P$", c=self.PHP_Color)
        plt.ylim([0.9999, 1])
        plt.yticks([0.9999, 1], ["0.9999", "1"])
        plt.xlim([0,1.5])
        plt.legend()
        plt.show()

    #Plot pi rate
    def Paper_Plot_Pi_Rate_Plot(self, Res=20):
        eta_set = np.linspace(0, 1.5, Res)
        deta = eta_set[1] - eta_set[0]

        H1_Set = []
        H12_Set = []
        PHP_Set = []
        h10_Set = []

        for eta in eta_set:

            H1 = np.round(self.H(self.mmult, eta, 1),10)
            H1_Val, H1_Vec = np.linalg.eigh(np.array(H1))
            H1_Vec = H1_Vec.transpose()
            pn = self.CG_Field_Momentum(self.mmult, 1, eta)
            H1_Set.append(np.abs(np.dot(np.conj(H1_Vec[1]), np.dot(pn, H1_Vec[0])))**2)

            H12 = np.round(self.H2(eta, 1),10)
            H12_Val, H12_Vec = np.linalg.eigh(np.array(H12))
            H12_Vec = H12_Vec.transpose()
            pn2 = self.CG_Field_Momentum(2, 1, eta)
            H12_Set.append(np.abs(np.dot(np.conj(H12_Vec[1]), np.dot(pn2, H12_Vec[0])))**2)

            PHP = np.round(self.PHP(eta, 1),10)
            PHP_Val, PHP_Vec = np.linalg.eigh(np.array(PHP))
            PHP_Vec = PHP_Vec.transpose()
            PHP_Set.append(np.abs(np.dot(np.conj(PHP_Vec[1]), np.dot(pn2, PHP_Vec[0])))**2)

            h10 = np.round(self.h10(eta),10)
            h10_Val, h10_Vec = np.linalg.eigh(np.array(h10))
            h10_Vec = h10_Vec.transpose()
            pn3 = self.CG_Field_Momentum(2, 0, eta)
            h10_Set.append(np.abs(np.dot(np.conj(h10_Vec[1]), np.dot(pn3, h10_Vec[0])))**2)
            

        plt.plot(eta_set, H1_Set, label="Bound", c=self.H1_Color)
        plt.plot(eta_set, H12_Set, label="$H_1^{(2)}$", c=self.H12_Color)
        plt.plot(eta_set, PHP_Set, label="$P H_1 P$", c=self.PHP_Color)
        plt.plot(eta_set, h10_Set, label="$h_1(0)$", c=self.h10_Color)
        plt.ylim([0, 1])
        plt.xlim([0,max(eta_set)])
        plt.legend()
        plt.show()







