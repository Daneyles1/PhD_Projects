import numpy as np
from pathlib import Path

class Read_Write_Class:

    # Check if a file exists at the given path.
    @staticmethod
    def Check_File_Exists(file_path):
        path = Path(file_path)
        return path.is_file()

    # Write a pair of floating-point numbers to a file.
    @staticmethod
    def Write_To_File(file_path, x, y):
        with open(file_path, 'a') as file:
            file.write(f"{x},{y}\n")

    # Read a pair of floating-point numbers from a file.
    @staticmethod
    def Read_From_File(file_path):
        if not Read_Write_Class.Check_File_Exists(file_path):
            return []
        
        with open(file_path, 'r') as file:
            line = file.readline().strip()
            x_str, y_str = line.split(',')
            return float(x_str), float(y_str)
        
    # Check if an element already exists in a set.
    @staticmethod
    def Check_Data_Already_Exists(x_Set:set, x_Element:float):
        return x_Element in x_Set

    # Read 2D data from a CSV file into a NumPy array.
    @staticmethod
    def Read_In_2D_Data(Filename):
        Data = []
        if Read_Write_Class.Check_File_Exists(Filename) == True:
            Data = np.loadtxt(Filename, delimiter=',', dtype='float')
        else:
            print("File Not Found")
        return Data

    # Write 2D data from a NumPy array to a CSV file.
    @staticmethod
    def Write_2D_Data(Filename, Data):
        np.savetxt(Filename, Data, delimiter=',')