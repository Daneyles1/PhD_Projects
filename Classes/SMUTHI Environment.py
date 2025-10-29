import sys
sys.path.insert(1, "Classes")

import smuthi.postprocessing.graphical_output as go
import smuthi.postprocessing.far_field as fd
import smuthi.postprocessing.far_field
import smuthi.initial_field
import smuthi.utility.cuda
import smuthi.simulation
import smuthi.particles
import smuthi.layers

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import constants as const

from Unit_Converter_Classes import *


class Smuthi_Environment: # Contain all information regarding a Environment setup in SMUTHI

    norm = 1/(8.85e-39)
    Solver = "LU"
    GMRES_Tolerance = 1e-1
    neff_imag = 1e-2
    neff_resolution = 1e-2
    Z = Distance_Class(0, "Nat")
    Previous_Sim = None #Track the most recent Simulation Run
    Run = True #Determine whether a new run is needed
    show = False
    MetaSurface = "FS"
    Layers = smuthi.layers.LayerSystem([0,0], [1,1])
    Basic_Layers = smuthi.layers.LayerSystem([0,0], [1,1])
    scatterers = []
    fixing_Value = -100

    @staticmethod
    def How_To_Use():
        print(""" This class aims to contain the key calculations using SMUTHI: 
              - When initiating the object, the class will represent a free space environment
              - To use specific envronment, use one of the "Generate_" functions 
              - To improve accuracy of predictions: 
                    1) up "Multipole" parameter in generate functions
                    2) decrease the "neff_imag" and "neff_resolution" class variables
              - The functions which should be used by the user are those names "Calculate_" 
              - To obtian function plots, use the functions which are named "Plot_"
              - The previous simulation is saved in the "Previous_Sim" class variable
        """)

    def __init__(self):
        pass

    def Format_E_field(self, E): #Reformat the electric field SMUTHI outputs
        return np.array([E[0][0], E[1][0], E[2][0]])

    def P0_Analytical(self, d: smuthi.initial_field.DipoleSource):
        omega = (2*np.pi*const.c/(d.vacuum_wavelength* 10**-9))
        dipole_mom = np.array(d.dipole_moment) / self.norm
        return (np.linalg.norm(dipole_mom)**2 / (12*np.pi)) * (omega**4 / (const.epsilon_0 * const.c**3))

    @staticmethod
    def Check_Not_Overlapping_Emitters(R1, R2):
        Overlap = False
        if R1[0] == R2[0] and R1[1] == R2[1] and R1[2] == R2[2]:
            Overlap = True
        return Overlap

#Run Key computations
    def Run_Simulation(self, d1:smuthi.initial_field.DipoleSource):
        simulation_A = None
        if self.Previous_Sim == None or self.Run == True:
            if len(self.scatterers) != 0:
                simulation_A = smuthi.simulation.Simulation(layer_system=self.Layers, particle_list=self.scatterers, 
                                            initial_field=d1, length_unit='nm', log_to_terminal=self.show, solver_type=self.Solver,
                                            solver_tolerance=self.GMRES_Tolerance, store_coupling_matrix=True, 
                                            coupling_matrix_lookup_resolution=5, neff_imag=self.neff_resolution,
                                            neff_resolution=self.neff_resolution)
            else:
                simulation_A = smuthi.simulation.Simulation(layer_system=self.Layers, particle_list=[], 
                                            initial_field=d1, length_unit='nm', log_to_terminal=self.show, solver_type="LU",)
            done = False
            while done==False:
                try:
                    simulation_A.run()
                    done = True
                except:
                    print("Error Running Sim... Trying again")
            self.Previous_Sim = simulation_A
        else:
            simulation_A = self.Previous_Sim
        return simulation_A

    def Run_Total_E_Field_Calculaion(self, Simulation:smuthi.simulation.Simulation, Pos_E:Distance_Class):
        pos2 = Pos_E.Nano
        done = False
        E12_init = None
        E12_scatt = None
        while done==False:
            try:
                E12_init = self.Format_E_field(go.compute_near_field(Simulation, pos2[0], pos2[1], pos2[2], type= 'initl'))
                done = True
            except:
                print("Error Running Init Field... Trying again")
        done = False
        while done==False:
            try:
                E12_scatt = self.Format_E_field(go.compute_near_field(Simulation, pos2[0], pos2[1], pos2[2], type= 'scatt'))
                done = True
            except:
                print("Error Running Scatt Field... Trying again")
        E12 = E12_init + E12_scatt
        return Electric_Field_Class(E12, "SI")

    def Run_Power_Dissipated(self, d1:smuthi.initial_field.DipoleSource):
        done = False
        P_ratio = None
        while done == False:
            try:
                pow = d1.dissipated_power(self.scatterers, self.Layers, self.show)
                pow_FS = d1.dissipated_power_homogeneous_background(self.Basic_Layers)
                P_ratio = pow/pow_FS
                done = True
            except Exception as e: 
                print(e)
                print("Error Running Power... Trying again")
        #if P_ratio[0] < 0:
        #    raise Exception("Negative Power Dissipated")
        return Power_Diss_Class(P_ratio[0] * self.P0_Analytical(d1), "SI")

#Rates      
    def Calculate_Rate_From_Power(self, d1:smuthi.initial_field.DipoleSource, omega:Frequency_Class):
        P = self.Run_Power_Dissipated(d1).SI
        G = (2*P/(omega.SI**3 * const.mu_0))
        Gamma = (2 /(const.epsilon_0*const.hbar * const.c**2)) * omega.SI**2 * G   
        return Frequency_Class(Gamma, "SI")

    def Calculate_Rate_From_E_Field(self, E:Electric_Field_Class, d:Dipole_Moment_Class, omega:Frequency_Class):
        G = np.dot(np.conj(d.SI), E.SI)/(omega.SI**2 * const.mu_0)
        Gamma = (2 /(const.epsilon_0*const.hbar * const.c**2)) * omega.SI**2 * np.imag(G)
        Delta = -1*(1 /(const.epsilon_0*const.hbar * const.c**2)) * omega.SI**2 * np.real(G)
        return Frequency_Class(Gamma, "SI"), Frequency_Class(Delta, "SI")

    def Calculate_Rate_Matrices(self, Positions:Distance_Class, Moments:Dipole_Moment_Class, wavelength:Distance_Class,
                                 Gamma_in:Frequency_Class=Frequency_Class([], "SI"), Delta_in:Frequency_Class=Frequency_Class([], "SI")):
        N = len(Positions.Nano)
        omega = Frequency_Class((2*np.pi*const.c/(wavelength.SI)), "SI")
        Delta = np.full([N,N], self.fixing_Value, dtype="float")
        Gamma = np.full([N,N], self.fixing_Value, dtype="float")
        if len(Gamma_in.Nat) != 0 and len(Delta_in.Nat) != 0:
            for i in range(N):
                for j in range(N):
                    Delta[i][j] = Delta_in.SI[i][j]
                    Gamma[i][j] = Gamma_in.SI[i][j]
        for i in range(N):
            Done = True
            for j in range(N):
                if Gamma[i][j] == self.fixing_Value or Delta[i][j] == self.fixing_Value:
                    Done = False
            if Done == False:
                d1 = smuthi.initial_field.DipoleSource(wavelength.Nano, Moments.SI[i]*self.norm, Positions.Nano[i])
                simulation_A = self.Run_Simulation(d1)
                self.Run = False
                for j in range(N):
                    if N != 1 and self.show == True:
                        print(str(j+1) + "/" + str(N) + ", " + str(i+1) + "/" + str(N))
                    if Gamma[i][j] == self.fixing_Value or Delta[i][j] == self.fixing_Value:
                        if i == j: #Diagonal Terms
                            Gamma[i][i] = self.Calculate_Rate_From_Power(d1, omega).SI
                            Delta[i][i] = 0
                        if j > i:
                            if Smuthi_Environment.Check_Not_Overlapping_Emitters(Positions.Nat[i], Positions.Nat[j]) == False:
                                pos2 = Distance_Class(Positions.Nano[j], "Nano")
                                d2 = Dipole_Moment_Class(Moments.SI[j], "SI")
                                E12 = self.Run_Total_E_Field_Calculaion(simulation_A, pos2)
                                Gam, Del = self.Calculate_Rate_From_E_Field(E12, d2, omega)
                                Gamma[i][j] = Gam.SI
                                Gamma[j][i] = np.conj(Gam.SI) #Using γij = γji^*
                                Delta[i][j] = Del.SI
                                Delta[j][i] = np.conj(Del.SI)
                            else:
                                Gamma[i][j] = self.Calculate_Rate_From_Power(d1, omega).SI
                                Delta[i][j] = 0
                                Gamma[j][i] = np.conj(Gamma[i][j]) #Using γij = γji^*
                                Delta[j][i] = 0
                    else:
                        if self.show == True:
                            print("Already computed")
                self.Run = True
        return Frequency_Class(Gamma, "SI"), Frequency_Class(Delta, "SI")

    def Calculate_Rate_Component(self, Positions:Distance_Class, Moments:Dipole_Moment_Class, wavelength:Distance_Class):
        omega = Frequency_Class((2*np.pi*const.c/(wavelength.SI)), "SI")
        d1 = smuthi.initial_field.DipoleSource(wavelength.Nano, Moments.SI[0]*self.norm, Positions.Nano[0])
        simulation_A = self.Run_Simulation(d1)
        Gamma = 0
        Delta = 0
        if Smuthi_Environment.Check_Not_Overlapping_Emitters(Positions.Nat[0], Positions.Nat[1]) == False:
            pos2 = Distance_Class(Positions.Nano[1], "Nano")
            d2 = Dipole_Moment_Class(Moments.SI[1], "SI")
            E12 = self.Run_Total_E_Field_Calculaion(simulation_A, pos2)
            Gamma, Delta = self.Calculate_Rate_From_E_Field(E12, d2, omega)
        else:
            Gamma = self.Calculate_Rate_From_Power(d1, omega)
            Delta = Frequency_Class(0, "SI")
        return Gamma, Delta
    
#Far Field
    def Find_Nearest_Angle_index(self, angle, angle_set):
        best_index = 0
        best_seperation = 10
        for i in range(len(angle_set)):
            sep = abs(angle - angle_set[i])
            if sep < best_seperation:
                best_index = i
                best_seperation = sep
        return best_index

    def Calculate_Greens_Far_Field_Far(self, Positions:Distance_Class, Moments:Dipole_Moment_Class, wavelength:Distance_Class, Polar1, Polar2, Azim1, Azim2):
        N = len(Positions.Nano)
        unit_Vector_1 = [np.sin(Polar1)*np.cos(Azim1), np.sin(Polar1)*np.sin(Azim1), np.cos(Polar1)]
        unit_Vector_2 = [np.sin(Polar2)*np.cos(Azim2), np.sin(Polar2)*np.sin(Azim2), np.cos(Polar2)]
        k0 = 2*np.pi/wavelength.Nat
        Far_1_Array = []
        Far_1_Matrix = np.zeros([N, N], dtype="complex_")
        Far_2_Array = []
        Far_2_Matrix = np.zeros([N, N], dtype="complex_")
        for i in range(N):
            d1 = smuthi.initial_field.DipoleSource(wavelength.Nano, Moments.SI[i]*self.norm, Positions.Nano[i])
            simulation_A = self.Run_Simulation(d1)
            far_field = fd.total_far_field(d1, self.scatterers, self.Layers) 
            E = np.array(far_field[0].electric_field_amplitude())
            Polar_Index = self.Find_Nearest_Angle_index(Polar1, far_field[0].polar_angles)
            Azim_Index = self.Find_Nearest_Angle_index(Azim1, far_field[0].azimuthal_angles)
            vec = [E[0][Polar_Index][Azim_Index], E[1][Polar_Index][Azim_Index], E[2][Polar_Index][Azim_Index]]
            Far_1_Array.append(vec) #* np.exp(-1j * k0 * np.dot(Positions.Nat[i], unit_Vector_1)))

            if Polar1 != Polar2 or Azim1 != Azim2:
                print("Seperate Directions")
                Polar_Index = self.Find_Nearest_Angle_index(Polar2, far_field[0].polar_angles)
                Azim_Index = self.Find_Nearest_Angle_index(Azim2, far_field[0].azimuthal_angles)
                vec = [E[0][Polar_Index][Azim_Index], E[1][Polar_Index][Azim_Index], E[2][Polar_Index][Azim_Index]]
                Far_2_Array.append(vec) #* np.exp(-1j * k0 * np.dot(Positions.Nat[i], unit_Vector_2)))
            else:
                Far_2_Array.append(vec)

        for i in range(N):
            for j in range(N):
                Far_1_Matrix[i][j] = np.dot(Far_1_Array[i], np.conj(Far_1_Array[j]))
                Far_2_Matrix[i][j] = np.dot(Far_2_Array[i], np.conj(Far_2_Array[j]))
        return Far_1_Matrix, Far_2_Matrix

#Spectral Density Calculations
    def Calculate_Spectral_Density(self, wavelength:Distance_Class, Moment:Dipole_Moment_Class, Position:Distance_Class):
        J = 0
        if wavelength.Nat == math.inf:
            J = Frequency_Class(0, "SI")
        else:
            d1 = smuthi.initial_field.DipoleSource(wavelength.Nano, Moment.SI*self.norm, Position.Nano)
            self.Run_Simulation(d1)
            P = self.Run_Power_Dissipated(d1).SI
            omega = Frequency_Class(2*np.pi/(wavelength.Nat), "Nat")
            J = Frequency_Class(2/(const.hbar*omega.SI*np.pi)*P, "SI")
        return J

    def Calculate_Cross_Spectral_Density(self, wavelength:Distance_Class, Rs:Distance_Class, ds:Dipole_Moment_Class):
        #simulation_A = self.Run_Simulation(d1)
        J_Cross = None
        if wavelength == 0:
            J_Cross = Frequency_Class(0, "SI")
        else:
            G_Cross, D_Cross = self.Calculate_Rate_Component(Rs, ds, wavelength)
            J_Cross = (1/(2*np.pi)) * G_Cross.Nat
        return Frequency_Class(J_Cross, "Nat")

#Preset Metasurface Definitions
    def Generate_Planar_Waveguide(self):
        h = 75
        self.Layers = smuthi.layers.LayerSystem(thicknesses=[0, h, 0],
                                         refractive_indices=[1, 3.5, 1])
        self.scatterers = []
        self.Z = Distance_Class(h, 'Nano')
        self.MetaSurface = "Slab"

    def Generate_Diego_ED_BIC(self, Multipole, N):
        R = 100
        a = 400
        n_Scatt = 3.5
        self.Layers = smuthi.layers.LayerSystem([0,0], [1,1])
        Scatterers = []
        for xi in range(-N, N+1):
            for yi in range(-N, N+1):
                Position = [xi*a, yi*a, 0]
                Scatterers.append(smuthi.particles.Sphere(Position, n_Scatt, R, Multipole, Multipole))
        #print(np.sqrt(len(Scatterers)))
        self.scatterers = Scatterers
        self.Z = Distance_Class(100, "Nano")
        self.MetaSurface = "Diego_BIC"

    def Generate_Incomplete_Diego_ED_BIC(self, Multipole, N, indexes = []):
        R = 100
        a = 400
        n_Scatt = 3.5
        self.Layers = smuthi.layers.LayerSystem([0,0], [1,1])
        Scatterers = []

        for xi in range(-N, N+1):
            for yi in range(-N, N+1):
                Position = [xi*a, yi*a, 0]
                Fill = True
                for index in indexes:
                    if len(index) >0:
                        if index[0] == xi and index[1] == yi:
                            Fill = False
                if Fill == True:
                    Scatterers.append(smuthi.particles.Sphere(Position, n_Scatt, R, Multipole, Multipole))
        
        self.scatterers = Scatterers
        self.Z = Distance_Class(100, "Nano")
        self.MetaSurface = "Diego_BIC"

    def Generate_Single_Sphere(self, Multipole=1, R=200, n=3.5):
        self.Layers = smuthi.layers.LayerSystem([0,0], [1,1])
        Scatterers = []
        Scatterers.append(smuthi.particles.Sphere([0,0,0], n, R, Multipole, Multipole))
        self.scatterers = Scatterers
        self.Z = Distance_Class(R, "Nano")
        self.MetaSurface = "1_Sphere" + str(R)

    def Generate_Tony_MD_BIC(self, Multipole, N):
        h = 150
        R = 130
        a = 310
        n_Substrate = 1.5
        n_cylinder = 2.5
        n_hBn = 2.13
        self.Layers = smuthi.layers.LayerSystem([0, h, 8, 0], [n_Substrate, 1, n_hBn, 1])
        Scatterers = []
        for xi in range(-N, N+1):
            for yi in range(-N, N+1):
                Position = [xi*a, yi*a, h/2]
                Scatterers.append(smuthi.particles.FiniteCylinder(position=Position,
                                                                  cylinder_radius=R, 
                                                                  refractive_index=n_cylinder,
                                                                  cylinder_height=h,
                                                                  l_max=Multipole, 
                                                                  m_max=Multipole))
        self.scatterers = Scatterers
        self.Z = Distance_Class(h,"Nano")
        self.MetaSurface = "Tony_BIC"

    def Generate_Tony_Substate(self):
        n_Substrate = 1.5
        n_hBn = 2.13
        self.Layers = smuthi.layers.LayerSystem([0, 8, 0], [n_Substrate, n_hBn, 1])
        self.scatterers = []
        self.Z = Distance_Class(4,"Nano")

    def Generate_Sb2_S3_Slab(self, h:Distance_Class, n):
        self.Layers = smuthi.layers.LayerSystem([0, h.Nano, 0], [1, n, 1])
        self.scatterers = []
        self.z = h


#Obtain Plots of Field
    def Plot_Far_Field(self, d:smuthi.initial_field.DipoleSource):
        simulation = self.Run_Simulation(d)
        #far_field = fd.total_far_field(d, self.scatterers, self.Layers)
        #go.show_total_far_field(far_field[0], show_plots=False, save_plots=True,)
        go.show_total_far_field(simulation, show_plots=False, save_plots=True)

    def Plot_Scatterer_Field(self, wavelength:Distance_Class, Moments:Dipole_Moment_Class, Positions:Distance_Class):
        d1 = smuthi.initial_field.DipoleSource(wavelength.Nano, Moments.SI[0]*self.norm, Positions.Nano[0])
        sim = self.Run_Simulation(d1)
        go.show_near_field(simulation=sim, quantities_to_plot={"norm(E_scat)", "E_scat_y"}, show_plots=False, save_plots=True, 
                           xmin=-1000, xmax=1000, ymin=-1000, ymax=1000, zmin=104, zmax=104, resolution_step=10,
                            save_data=True, data_format='ascii',
                            outputdir="/Users/user/Desktop/Code/Python Scripts/BIC_Work/Data_Files/Scattered_Files/")

    def Plot_Scatterers(self):
        fig, ax = plt.subplots()
        x_max = 10
        for sph in self.scatterers:
            x, y, z = sph.position
            if x > x_max:
                x_max = x
            circle = plt.Circle((x, y), self.Z.Nano, color='steelblue', alpha=0.6)
            ax.add_patch(circle)
        ax.set_aspect('equal')
        ax.set_xlim([-1.2*x, 1.2*x])
        ax.set_ylim([-1.2*x, 1.2*x])
        plt.show()
