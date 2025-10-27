import numpy as np
from scipy import constants as const

#Class to convert between Natural units and S.I. units
class NAT_SI_conversion: 
    Nat_to_Hertz = const.e/const.hbar
    Hertz_to_Nat = 1/Nat_to_Hertz

    Nat_to_Second = Hertz_to_Nat
    Second_to_Nat = Nat_to_Hertz

    Nat_to_Kelvin = const.k/const.e
    Kelvin_to_Nat = 1/Nat_to_Kelvin

    Nat_to_Metre = const.hbar*const.c/const.e
    Metre_to_Nat = 1/Nat_to_Metre

    Nat_to_Inv_Metre = Metre_to_Nat
    Inv_Metre_to_Nat = Nat_to_Metre

    Nat_to_Coulomb = const.e/np.sqrt(4*np.pi*const.alpha)
    Coulomb_to_Nat = 1/Nat_to_Coulomb

    Nat_to_Newton = (const.e**2)/(const.hbar*const.c)
    Newton_to_Nat = 1/Nat_to_Newton

    Nat_to_kg = const.e/(const.c**2)
    kg_to_Nat = 1/Nat_to_kg

#Class to hold a distance value or array to access the diffent unit formats
class Distance_Class:

    def __init__(self, Distance, units:str):
        if type(Distance) != list:
            if units.lower() == "nm" or units.lower() == "nano":
                self.Nano = Distance
                self.SI = Distance * (10**-9)
                self.Nat = self.SI * NAT_SI_conversion.Metre_to_Nat
            elif units.lower() == "nat":
                self.Nat = Distance
                self.SI = Distance * NAT_SI_conversion.Nat_to_Metre
                self.Nano = self.SI * (10**9)
            elif units.lower() == "si" or units.lower() == "metre":
                self.SI = Distance
                self.Nano = Distance * (10**9)
                self.Nat = self.SI * NAT_SI_conversion.Metre_to_Nat
        elif type(Distance) == list:
            if units.lower() == "nm" or units.lower() == "nano":
                self.Nano = np.array(Distance)
                self.SI = np.array(Distance) * (10**-9)
                self.Nat = self.SI * NAT_SI_conversion.Metre_to_Nat
            elif units.lower() == "nat":
                self.Nat = np.array(Distance)
                self.SI = np.array(Distance) * NAT_SI_conversion.Nat_to_Metre
                self.Nano = self.SI * (10**9)
            elif units.lower() == "si" or units.lower() == "metre":
                self.SI = np.array(Distance)
                self.Nano = np.array(Distance) * (10**9)
                self.Nat = self.SI * NAT_SI_conversion.Metre_to_Nat
        else:
            print("Error")

#Class to hold a angulat frequency value or array to access the diffent unit formats
class Frequency_Class:
    def __init__(self, Freq, unit:str):
        if type(Freq) == list:
            Freq = np.array(Freq)
        if unit.lower() == "si":
            self.SI = Freq
            self.Nano = self.SI * (10**9)
            self.Nat = self.SI * NAT_SI_conversion.Hertz_to_Nat            
        elif unit.lower() == "nano":
            self.SI = Freq * (10**-9)
            self.Nano = self.SI 
            self.Nat = self.SI * NAT_SI_conversion.Hertz_to_Nat   
        elif unit.lower() == "nat":
            self.Nat = Freq
            self.SI = self.Nat * NAT_SI_conversion.Nat_to_Hertz
            self.Nano = self.SI * (10**9)                    

#Class to hold a dipole moment (Cm) value or array to access the diffent unit formats
class Dipole_Moment_Class:

    def __init__(self, Moment, unit:str):
        if unit.lower() == "si":
            self.SI = np.array(Moment)
            self.Nat = np.array(Moment) * NAT_SI_conversion.Coulomb_to_Nat * NAT_SI_conversion.Metre_to_Nat 
        elif unit.lower() == "nat":
            self.Nat = np.array(Moment)
            self.SI = np.array(Moment) * NAT_SI_conversion.Nat_to_Coulomb * NAT_SI_conversion.Nat_to_Metre

#Class to hold a inverse distance value or array to access the diffent unit formats
class Inv_Metre_Class:

    def __init__(self, Inv_Metre, unit:str):
        if type(Inv_Metre) == list:
            Inv_Metre = np.array(Inv_Metre)
        if unit.lower() == "nat":
            self.Nat = Inv_Metre
            self.SI = Inv_Metre * NAT_SI_conversion.Nat_to_Inv_Metre
        elif unit.lower() == "si":
            self.Nat = Inv_Metre * NAT_SI_conversion.Inv_Metre_to_Nat
            self.SI = Inv_Metre

#Class to hold a Electric field value or array to access the diffent unit formats
class Electric_Field_Class:
    def __init__(self, E, unit:str):
        if unit.lower() == "si":
            self.SI = np.array(E)
            self.Nat = np.array(E) * NAT_SI_conversion.Newton_to_Nat / NAT_SI_conversion.Coulomb_to_Nat
        elif unit.lower() == "nat":
            self.Nat = np.array(E)
            self.SI = np.array(E) * NAT_SI_conversion.Nat_to_Newton / NAT_SI_conversion.Nat_to_Coulomb

#Class to hold a Power value or array to access the diffent unit formats
class Power_Diss_Class:
    def __init__(self, P, unit:str):
        if unit.lower() == "si":
            self.SI = P
            self.Nat = P * NAT_SI_conversion.Joules_to_Nat * NAT_SI_conversion.Hertz_to_Nat
        elif unit.lower() == "nat":
            self.Nat = P
            self.SI = P * NAT_SI_conversion.Nat_to_Joules * NAT_SI_conversion.Nat_to_Hertz

#Class to hold a Temperature value or array to access the diffent unit formats
class Temp_Class:
    def __init__(self, T, unit:str):
        if unit.lower() == "si":
            self.SI = T
            self.Nat = T * NAT_SI_conversion.Kelvin_to_Nat
        elif unit.lower() == "nat":
            self.Nat = T
            self.SI = T * NAT_SI_conversion.Nat_to_Kelvin















