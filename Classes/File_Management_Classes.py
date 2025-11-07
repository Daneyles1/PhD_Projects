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

        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)  # <-- create folders if missing
        with open(file_path, 'a+') as file:
            file.write(f"{np.round(x, 10)},{np.round(y, 10)}\n")

    # Read a pair of floating-point numbers from a file.
    @staticmethod
    def Read_From_File(file_path):
        if not Read_Write_Class.Check_File_Exists(file_path):
            return []
        
        result = []
        with open(file_path, 'r') as file:
            for line in file:
                if line.strip():  # Skip empty lines
                    x_str, y_str = line.strip().split(',')
                    result.append((float(x_str), float(y_str)))
        return result
        
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

    @staticmethod
    def Sort_Both_Lists(x, y):
        sorted_pairs = sorted(zip(x, y), key=lambda pair: pair[0])
        x_sorted, y_sorted = map(list, zip(*sorted_pairs))
        return x_sorted, y_sorted
    
    @staticmethod
    def Append_to_1D_Data(Filename, Data):
        with open(Filename, "a+") as f:
            f.write(str(Data)+"\n")

    @staticmethod
    def Read_In_1D_Data(Filename):
        data = []
        data_new = []
        if Read_Write_Class.Check_File_Exists(Filename) == True:
            with open(Filename, "r+") as f:
                data = f.readlines()
            for d in data:
                if d.replace("/n",'') != '':
                    data_new.append(float(d.replace("\n",'')))
        else:
            print("File Not Found")
        return data_new








