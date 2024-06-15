import os
import numpy as np
import sys

import instrumentor as inst
import random

from deap import creator, base, tools, algorithms

import shutil

import json
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import wilcoxon

# ------------------------ HYPERPARAMETERS ------------------- #

# Constants for random string generation
MIN_INT = -1_000
MAX_INT = 1_000
MAX_STRING_LENGTH = 10
POOL_SIZE = 1_000


# Number of copies of the test archive to generate
COPIES = 10

# ------------------------ Functions ------------------------ #

# Function to get the path files in the "parsed" folder
def parsed_paths():
    list_of_files = []
    files_in_benchmark = "parsed"
    
    # Iterate over files in the "parsed" folder
    for filename in os.listdir(files_in_benchmark):
        f = os.path.join(files_in_benchmark, filename)
        
        # checking if it is a file
        if os.path.isfile(f):
            list_of_files.append(f)
    
    return list_of_files


# Function to get the codes from each parsed file
def parsed_codes(list_of_files):
    list_of_codes = []
    dict_codes = {file: [] for file in list_of_files}
    
    # Iterate over paths in the list_of_files
    for path in list_of_files:
        with open(path, 'r') as file:
            code = file.read()
            list_of_codes.append(code)
            dict_codes[path] = code
    
    return list_of_codes, dict_codes


# Function to extract function signatures from code in each parsed file
def parsed_signatures(my_dict):
    
    # Iterate over files in the dictionary
    for idx in my_dict:
        lines = my_dict[idx].split("\n")
        signature = ""
        
        # Iterate over lines in the code
        for line in lines:
            if line.startswith("def"):
                if signature != "":
                    signature += "\n new_signature " + line
                else:
                    signature += line
            if signature != "":
                my_dict[idx] = signature
    
    return my_dict


# Function to get the types of parameters from each signature
def get_types_of_parameters(my_dict):
    
    # Iterate over files in the dictionary
    for idx in my_dict:
        signatures = my_dict[idx]
        types = []
        
        # Iterate over signatures in the file
        for signature in signatures:
            
            start = signature.index("(") + 1
            end = signature.index(")")
            # Extract substring between '(' and ')' which contains parameters
            substring = signature[start: end]
            # print(f"substring is {substring}")
            # Split parameters by comma
            split_by_comma = substring.split(",")
            # print(f"split by comma is {split_by_comma}")
            
            tipos = []
            # Iterate over arguments in the signature
            for argument in split_by_comma:
                split_by_twoDots = argument.split(":")
                tipo = split_by_twoDots[-1]
                tipos.append(tipo)
            types.append((signature, tipos))
        
        my_dict[idx] = types
    
    return my_dict


# ------------------------ Random Variables ------------------------ #

# Function to initialize a random string variable
def init_str_variable():
    
    # Generate a random length for the string
    length = np.random.randint(0, MAX_STRING_LENGTH)
    sentence = ""
    
    # Generate a random string of lowercase characters
    for _ in range(length):
        sentence += chr(np.random.randint(97, 122))
    
    return sentence


# Function to initialize a random integer variable
def init_int_variable():
    return np.random.randint(MIN_INT, MAX_INT)


# Function to initialize key-value pairs with a random string and integer
def init_key_value_pairs():
    return (init_str_variable(), init_int_variable())


# Function to generate the pool of random strings
def single_string():
    sentence = ""
    sentences = []
    for _ in range(POOL_SIZE):
        sentence = init_str_variable()
        sentences.append(sentence)
    
    return sentences


# Function to generate the pool of random integers
def single_int():
    int_variables = []
    for _ in range(POOL_SIZE):
        int_variables.append(init_int_variable())
    
    return int_variables


# Function to generate the pool of random (strings, integers)
def double_string():
    sentences = []
    for _ in range(POOL_SIZE):
        sentences.append((init_str_variable(), init_str_variable()))
    
    return sentences


# Function to generate the pool of random (integers, integers)
def double_int():
    int_variables = []
    for _ in range(POOL_SIZE):
        int_variables.append((init_int_variable(), init_int_variable()))
    
    return int_variables


# Function to generate the pool of random strings
def string_int():
    key_value_pairs = []
    for _ in range(POOL_SIZE):
        key_value_pairs.append(init_key_value_pairs())
    
    return key_value_pairs


# Function to generate the pool of random (integers, integers, integers)
def triple_int():
    int_variables = []
    for _ in range(POOL_SIZE):
        int_variables.append(
            (init_int_variable(), init_int_variable(), init_int_variable()))
    
    return int_variables

# ------------------------ Create Pools ------------------------ #

# Function to create a pool of variables based on the provided element types
def create_pool(elem):
    # Check the length of the element tuple
    if len(elem) == 1:
        x = elem[0]
        
        # If there is a single element and it's "str", generate a pool of random strings
        # Otherwise, generate a pool of random integers
        if x == "str":
            variables = single_string()
        else:
            variables = single_int()
    elif len(elem) == 2:
        x, y = elem[0], elem[1]
        
        # If there are two elements and they are both "str", generate a pool of pairs of random strings
        # If the first element is "str" and the second is "int", generate a pool of key-value pairs with random strings and integers
        # Otherwise, generate a pool of pairs of random integers
        if x == "str" and y == "str":
            variables = double_string()
        elif x == "str" and y == "int":
            variables = string_int()
        else:
            variables = double_int()
    else:
        # If there are more than two elements, generate a pool of triples of random integers
        variables = triple_int()
    
    return variables

#  ------------------------ Mutation ------------------------ #

# Function to mutate a single string element
def mutate_single_string(elem):
    
    # If the string is empty, no mutation is performed
    if len(elem) == 0:
        return elem
    else:
        # Randomly select an index and replace the character at that index with a new random lowercase character
        idx = np.random.randint(0, len(elem))
        return elem[:idx] + chr(np.random.randint(97, 122)) + elem[idx+1:]


# Function to mutate a single integer element
def mutate_single_int():
    return init_int_variable()


# Function to mutate a key-value pair (tuple) with a string key and integer value
def mutate_tuple_key_value(key, value):
    # If the key is empty, replace the value with a new random integer
    if len(key) == 0:
        new_value = init_int_variable()
        return (key, new_value)
    
    # Randomly select an index
    idx = np.random.randint(len(key))
    
    # Randomly decide whether to mutate the key or the value
    if np.random.uniform() < 0.5:
        # Mutate only the key
        new_key = key[:idx] + chr(np.random.randint(97, 122)) + key[idx+1:]
        return (new_key, value)
    else:
        # Mutate only the value
        new_value = init_int_variable()
        return (key, new_value)


# Function to mutate a tuple with two string elements
def mutate_tuple_str_str(str1, str2):
    # If both strings are empty, no mutation is performed
    if len(str1) == 0 and len(str2) == 0:
        return str1, str2
    
    # If one of the strings is empty, mutate the non-empty string
    if len(str1) == 0:
        idx2 = np.random.randint(0, len(str2))
        new_str2 = str2[:idx2] + \
            chr(np.random.randint(97, 122)) + str2[idx2+1:]
        return (str1, new_str2)
    elif len(str2) == 0:
        idx1 = np.random.randint(0, len(str1))
        new_str1 = str1[:idx1] + \
            chr(np.random.randint(97, 122)) + str1[idx1+1:]
        return (new_str1, str2)
    
    # If both strings are non-empty, randomly select indices and decide which string(s) to mutate
    idx1 = np.random.randint(0, len(str1))
    idx2 = np.random.randint(0, len(str2))
    
    if np.random.uniform() < 1/3:
        # Mutate only the first string
        new_str1 = str1[:idx1] + \
            chr(np.random.randint(97, 122)) + str1[idx1+1:]
        return (new_str1, str2)
    elif np.random.uniform() < 1/3:
        # Mutate only the second string
        new_str2 = str2[:idx2] + \
            chr(np.random.randint(97, 122)) + str2[idx2+1:]
        return (str1, new_str2)
    else:
        # Mutate both strings
        new_str1 = str1[:idx1] + \
            chr(np.random.randint(97, 122)) + str1[idx1+1:]
        new_str2 = str2[:idx2] + \
            chr(np.random.randint(97, 122)) + str2[idx2+1:]
        return (new_str1, new_str2)


# Function to mutate a tuple with two integer elements
def mutate_tuple_int_int(int1, int2):
    # Randomly decide which integer(s) to mutate
    if np.random.uniform() < 1/3:
        # Mutate only the first integer
        new_int1 = init_int_variable()
        return (new_int1, int2)
    elif np.random.uniform() < 1/3:
        # Mutate only the second integer
        new_int2 = init_int_variable()
        return (int1, new_int2)
    else:
        # Mutate both integers
        new_int1 = init_int_variable()
        new_int2 = init_int_variable()
        return(new_int1, new_int2)


# Function to mutate a tuple with three integer elements
def mutate_tuple_int_int_int(int1, int2, int3):
    # Randomly choose a mutation case (1 to 6)
    case = np.random.randint(1, 7)
    
    # Perform the mutation based on the chosen case
    if case == 1:
        new_int1 = init_int_variable()
        return (new_int1, int2, int3)
    elif case == 2:
        new_int2 = init_int_variable()
        return (int1, new_int2, int3)
    elif case == 3:
        new_int3 = init_int_variable()
        return (int1, int2, new_int3)
    elif case == 4:
        new_int1 = init_int_variable()
        new_int2 = init_int_variable()
        return (new_int1, new_int2, int3)
    elif case == 5:
        new_int1 = init_int_variable()
        new_int3 = init_int_variable()
        return (new_int1, int2, new_int3)
    elif case == 6:
        new_int2 = init_int_variable()
        new_int3 = init_int_variable()
        return (int1, new_int2, new_int3)
    else:
        new_int1 = init_int_variable()
        new_int2 = init_int_variable()
        new_int3 = init_int_variable()
        return (new_int1, new_int2, new_int3)


# Function to perform mutation on an element based on its type
def mutation(elem):
    # Check the type of the element and perform the corresponding mutation
    if isinstance(elem, str):
        return mutate_single_string(elem)
    
    elif isinstance(elem, int):
        return mutate_single_int()
    
    else:
        a, b = elem[0], elem[1]
        # Check the types of the tuple elements and perform the corresponding mutation
        if isinstance(a, str) and isinstance(b, int):
            return mutate_tuple_key_value(a, b)
        
        elif isinstance(a, str) and isinstance(b, str):
            return mutate_tuple_str_str(a, b)
        
        else:
            # (int, int, [int]) handled here
            if len(elem) == 2:
                return mutate_tuple_int_int(a, b)
            else:
                return mutate_tuple_int_int_int(a, b, elem[2])


# ------------------------ Crossover ------------------------ #

# Function to perform crossover between two individuals
def crossover(individual1, individual2):
    
    # If both individuals are integers, swap their values
    if isinstance(individual1, int) and isinstance(individual2, int):
        return individual2, individual1
    
    # If either individual has length less than or equal to 1, no crossover is performed
    if len(individual1) <= 1 or len(individual2) <= 1:
        return individual1, individual2
    
    else:
        # Extract head and tail from the individual 1
        head1 = individual1[0]
        tail1 = individual1[1]
        
        # If the head of individual 1 is an integer
        if isinstance(head1, int):
            # Extract head and tail from individual 2
            head2 = individual2[0]
            tail2 = individual2[1]
            
            # If both individuals have a third element, extract it
            if len(individual1) == 3 and len(individual2) == 3:
                # Create offspring with head from individual 1 and tail from individual 2, preserving the third element
                offspring1 = (head1, tail2, individual1[2])
                # Create offspring with head from individual 2 and tail from individual 1, preserving the third element
                offspring2 = (head2, tail1, individual2[2])
                
                return offspring1, offspring2
            
            # Create offspring with head from individual 1 and tail from individual 2
            else:
                offspring1 = (head1, tail2)
                offspring2 = (head2, tail1)
                
                return offspring1, offspring2
        
        # If the head of individual 1 is a string and the tail is an integer
        elif isinstance(head1, str) and isinstance(tail1, int):
            # Extract head and tails from individual 2
            head2 = individual2[0]
            tail2 = individual2[1]
            
            # If either head1 or head2 is empty, no crossover is performed
            if not head1 or not head2:
                return individual1, individual2
            
            # Randomly select a position for crossover
            pos = np.random.randint(0, min(len(head1), len(head2)))
            
            # Create offspring with crossover between str (keys), preserving the original tails (int: values)
            offspring1 = head1[:pos] + head2[pos:]
            offspring2 = head2[:pos] + head1[pos:]
            
            # Update individuals with the new offspring
            individual1 = (offspring1, tail1)
            individual2 = (offspring2, tail2)
            
            return individual1, individual2
        
        else:  # strings
            
            # If either individual 1 or individual 2 is empty, no crossover is performed
            if not individual1 or not individual2:
                return individual1, individual2
            
            # Randomly select a position for crossover
            pos = np.random.randint(0, min(len(individual1), len(individual2)))
            
            # Create offspring with crossover between individual 1 and individual 2
            offspring1 = individual1[:pos] + individual2[pos:]
            
            # Create offspring with crossover between individual2 and individual1
            offspring2 = individual2[:pos] + individual1[pos:]
            
            return offspring1, offspring2


# ------------------------ Testing ------------------------ #
# NOT USED ANYMORE
def testing_function(def_dict, dict_codes):

    for file in def_dict:
        print(file)
        file_path = file
        function_names = def_dict[file][::2]
        variable_types = def_dict[file][1::2]
        print(function_names)
        print(variable_types)

        for idx, f in enumerate(function_names):
            args = create_pool(variable_types[idx])[:1][0]
            print(args)
            print(type(args))

            try:
                current_module = sys.modules[__name__]
                code = compile(dict_codes[file_path],
                               filename="<string>", mode="exec")
                exec(code, current_module.__dict__)
                dict_distance_true, dict_distance_false = inst.get_distance_dicts()
                print(f"true distance {dict_distance_true} and {dict_distance_false}")

                if isinstance(args, tuple):
                    x = globals()[f](*args)
                else:
                    x = globals()[f](args)
                    
                print(f"x is {x}")
                print(f"true distance {dict_distance_true} and {dict_distance_false}")
                
                dict_distance_true.clear()
                dict_distance_false.clear()
            
            except AssertionError as e:
                print(f"AssertionError {e} for function {f}")
                pass
            except BaseException as e:
                print(f"BaseError {e} for function {f}")
                raise e
        
        print("--------------------------------------------------")


# ------------------------ FUZZY ------------------------ #

# Function to execute a given function with arguments and return the result along with distance dictionaries
def execute_function(file, f, args):
    # Retrieve the current module
    current_module = sys.modules[__name__]
    # Compile the file
    code = compile(file, filename="<string>", mode="exec")
    # Execute the compiled code within the current module's namespace
    exec(code, current_module.__dict__)
    # Get distance dictionaries from the instrumentor module
    dict_distance_true, dict_distance_false = inst.get_distance_dicts()
    
    # Call the specified function with arguments
    if isinstance(args, tuple):
        y = globals()[f](*args)
    else:
        y = globals()[f](args)
    
    # If the result is a non-empty string, replace special characters
    if isinstance(y, str) and len(y) >= 1:
        y.replace('\\', '\\\\').replace('"', '\\"')
    
    return y, dict_distance_true, dict_distance_false


# Function to keep track of distances and arguments for true and false cases
def keep_track(true_archive, false_archive, dict_distance_true, dict_distance_false, args_true_archive, args_false_archive, args, y):
    
    # Update true_archive and args_true_archive based on dict_distance_true
    for idx in dict_distance_true:
        if idx not in true_archive:
            true_archive[idx] = dict_distance_true[idx]
            args_true_archive[idx] = (args, y)
        if dict_distance_true[idx] == 0:
            true_archive[idx] = dict_distance_true[idx]
            args_true_archive[idx] = (args, y)
    
    # Update false_archive and args_false_archive based on dict_distance_false
    for idx in dict_distance_false:
        if idx not in false_archive:
            false_archive[idx] = dict_distance_false[idx]
            args_false_archive[idx] = (args, y)
        if dict_distance_false[idx] == 0:
            false_archive[idx] = dict_distance_false[idx]
            args_false_archive[idx] = (args, y)
    
    return true_archive, false_archive, args_true_archive, args_false_archive


# Function to update the pool with mutated or crossover individuals
def update_pool(pool, args):
    # Randomly decide whether to add a mutation or crossover to the pool
    if np.random.uniform() < 1/3:
        pool.append(mutation(args))
    elif np.random.uniform() < 1/3:
        x, y = crossover(args, random.choice(pool))
        pool.append(x)
        pool.append(y)


# Function to write test cases based on the provided dictionary, value, and folder
def write_test(temp_dict, val="true", folder = "Fuzzy"):
    
    # Loop through keys (file names) in the dictionary
    for key in temp_dict:
        code = ""
        # Extract original file name from the key
        original_file_name = key.split("/")[1].split("_instrumented")[0]
        print(f"original file name {original_file_name}")
        tuples = temp_dict[key]
        
        # If the value is "true", generate import statements and class definition
        if val == "true":
            for elem in tuples:
                f_name = elem[0].split("_instrumented")[0]
                code += "from benchmark." + original_file_name+" import " + f_name + "\n"
            
            code += "from unittest import TestCase\n"
            
            code +="\nclass Test_" + original_file_name + "(TestCase):\n"
        
        # Loop through function names and their corresponding archives in tuples
        for f_name, arch in tuples:
            # Loop through branch numbers and their corresponding (x, y) values
            for branch_nr in arch.keys():
                x, y = arch[branch_nr]
                
                # Generate test function name
                test_f_name = f"test_{f_name}_{val}_{branch_nr}(self):\n"
                
                code += f"\tdef {test_f_name}"
                
                code += f"\t\ty = {f_name.split('_instrumented')[0]}"
                
                args = x
                
                # Handle argument formatting based on the folder type
                if folder == "Fuzzy":
                    if isinstance(args, int):
                        code += "(" + str(x) + ")\n"
                    elif isinstance(args, str):
                        code += '("'+str(x)+'")' + "\n"
                    else:
                        code += str(x) + "\n"
                else:
                    args_s="("
                    for arg in x:
                        if isinstance(arg, str):
                            arg = "'"+str(arg)+"'"
                        args_s += str(arg) + ","
                    args_s = args_s[:-1]
                    code += args_s+")\n"
                
                # Add the assert statement
                if isinstance(y, str):
                    code += "\t\tassert y==" + '"'+str(y)+'"'+"\n\n"
                else:
                    code += "\t\tassert y=="+str(y)+"\n\n"
                
        # Append or write the string code to the file in the specified folder
        file_name = folder+"/" +original_file_name+"_test.py"
        if val == "true":
            with open(file_name, 'w') as file:
                file.write(code)
        else:
            with open(file_name, 'a') as file:
                file.write("\n" + code)


def fuzzy_testing(def_dict, dict_codes):
    avg_pool = []
    
    args_true_list, args_false_list = [], []
    
    # Loop through each file in the dictionary    
    for file in def_dict:
        file_path = file
        function_names = def_dict[file][::2]
        variable_types = def_dict[file][1::2]
        
        args_true = []
        args_false = []
        
        # Iterate over function names and their corresponding variable types
        for idx, f in enumerate(function_names):
            true_archive, false_archive = None, None
            args_true_archive = {}
            args_false_archive = {}
            pool = create_pool(variable_types[idx])
            
            # Loop through different sets of arguments for the function
            for args in pool:
                try:
                    # Execute the function and get distances
                    y, dict_distance_true, dict_distance_false = execute_function(
                        dict_codes[file_path], f, args)
                    
                    # If the archives are empty, initialize them
                    if true_archive is None or false_archive is None:
                        true_archive = dict_distance_true.copy()
                        false_archive = dict_distance_false.copy()
                        
                        # keep track of the best args for op number
                        args_true_archive = true_archive.copy()
                        for p in args_true_archive:
                            args_true_archive[p] = (args, y)
                        
                        args_false_archive = false_archive.copy()
                        for q in args_false_archive:
                            args_false_archive[q] = (args, y)
                    
                    else:
                        # Update archives and keep track of the best arguments
                        true_archive, false_archive, args_true_archive, args_false_archive = keep_track(
                            true_archive, false_archive, dict_distance_true, dict_distance_false, args_true_archive, args_false_archive, args, y)
                    
                    # Clear dictionaries to avoid interference
                    dict_distance_true.clear()
                    dict_distance_false.clear()
                    
                    # Update the pool
                    update_pool(pool, args)
                    avg_pool.append(len(pool))
                
                except AssertionError as e:
                    # print(f"AssertionError {e} for function {f}")
                    pass
                except BaseException as e:
                    # print(f"BaseError {e} for function {f}")
                    raise e
            
            # Organize and store results for each function
            args_true.append((f, dict(sorted(args_true_archive.items()))))
            args_false.append((f, dict(sorted(args_false_archive.items()))))
        
        # Store results for each file
        args_true_list.append((file_path, args_true))
        args_false_list.append((file_path, args_false))
    
    # print(f"args true is {args_true_list}")
    # print(f"args false is {args_false_list}")
    # print(f"avg pool is {np.mean(avg_pool)}")
    # print("--------------------------------------------------")
    
    # Prepare data for writing test cases
    temp_dict_true = {}
    for elem in args_true_list:
        key = elem[0]
        if key not in temp_dict_true:
            temp_dict_true[key] = []
        temp_dict_true[key].append(elem[1])
    
    for key in temp_dict_true:
        tuples = temp_dict_true[key]
        for elem in tuples:
            temp_dict_true[key] = elem
    
    
    temp_dict_false = {}
    for elem in args_false_list:
        key = elem[0]
        if key not in temp_dict_false:
            temp_dict_false[key] = []
        temp_dict_false[key].append(elem[1])
    
    for key in temp_dict_false:
        tuples = temp_dict_false[key]
        for elem in tuples:
            temp_dict_false[key] = elem
    
    # Write test cases for true and false conditions
    write_test(temp_dict_true, val="true", folder = "Fuzzy")
    write_test(temp_dict_false, val="false", folder = "Fuzzy")


# ------------------------ FUZZY ------------------------ #

# Get the list of file paths in the "parsed" folder
list_of_files = parsed_paths()

# print(f"list of files is\n {list_of_files}")

# Create a dictionary to store the parsed code for each file
my_dict = {_: [] for _ in range(len(list_of_files))}

# Get the codes from each parsed file
list_of_codes, dict_codes = parsed_codes(list_of_files)

# print(f"list of codes is\n {list_of_codes}")

# print()

# print(f"dict of codes is\n {dict_codes}")

# for file in dict_codes:
#     print(f"\nfile {file}\n {dict_codes[file]}")

# Populate the dictionary with parsed codes
for idx, elem in enumerate(list_of_codes):
    my_dict[idx] = elem

# Extract function signatures from the parsed code
my_dict = parsed_signatures(my_dict)

# print(f"my dict is {my_dict}")

# Split the signatures into individual functions
# print(f"my dict is {my_dict}")
for idx in my_dict:
    elem = my_dict[idx]
    if "new_signature" in elem:
        splitted = elem.split("new_signature")
        splitted = [x.strip() for x in splitted]
        my_dict[idx] = splitted
    else:
        my_dict[idx] = [elem]
# print(f"my dict is {my_dict}")

# Extract types of parameters from function signatures
my_dict = get_types_of_parameters(my_dict)

# print()
# print()
# print()
# print(f"my dict is {my_dict}")
# Organize the extracted information
for idx in my_dict:
    array = my_dict[idx]
    new_tuple = ()
    for t in array:
        a, b = t[0], t[1]
        a = a.split("(")[0].split(" ")[1]
        b = [typ.strip() for typ in b]
        new_tuple += (a, b)
    my_dict[idx] = list(new_tuple)
# print()
# print(f"my dict is\n {my_dict}")

# Create a dictionary to store the definition of functions in each file
def_dict = {_: [] for _ in list_of_files}

# Populate the dictionary with the organized information
for idx, file in enumerate(def_dict):
    def_dict[file] = my_dict[idx]

# print()
# print(f"def dict is\n {def_dict}")
# print()

# for file in def_dict:
#     print(f"file {file}: {def_dict[file]}")

# Call the testing function with the dictionary and parsed codes
# testing_function(def_dict, dict_codes)

# Copy and paste the "Fuzzy" folder into the "Archive" folder multiple times
for i in range(1, COPIES + 1):
    print(f"copy {i}")
    if not os.path.exists('Archive/fuzzer_test_archive/tests_fuzzer_copy_' + str(i)):
        fuzzy_testing(def_dict, dict_codes)
        original_folder = 'Fuzzy'
        destination_folder = 'Archive/fuzzer_test_archive/tests_fuzzer_copy_' + str(i)
        shutil.copytree(original_folder, destination_folder)


# ------------------------ DEAP ------------------------ #

# Variables for storing branch information
my_branches = None
archive_true_branches: dict[int, str] = {}
archive_false_branches: dict[int, str] = {}

# HYPERPARAMETERS FOR DEAP
NPOP = 300  # 300
NGEN = 10
INDMUPROB = 0.05  # 0.05
MUPROB = 0.3  # 0.1
CXPROB = 0.3  # 0.5
TOURNSIZE = 3
LOW = -1000
UP = 1000
REPS = 1
MAX_STRING_LENGTH = 10

# Deap creators for fitness and individual
creator.create("Fitness", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.Fitness)

# # Function to create the individual for the GA as a tuple of variables
def create_individual():
    global current_arg
    lista = []
    for elem in current_arg:
        # based on the type of the variable, create a random variable
        if elem == "int":
            lista.append(init_int_variable())
        elif elem == "str":
            lista.append(init_str_variable())
    return tuple(lista)

# Function to normalize distance values
def normalize(x):
    return x / (1.0 + x)

# Function to calculate the fitness of the individual
def get_fitness(individual):
    
    # Extract variables from the individual
    x = individual[0]
    
    # Reset any distance values from previous executions
    global distances_true, distances_false
    global branches, archive_true_branches, archive_false_branches
    global current_f
    global my_branches
    
    # Retrieve and clear distance dictionaries
    distances_true, distances_false = inst.get_distance_dicts()
    distances_true.clear()
    distances_false.clear()
    
    try:
        # Run the function under test with the provided arguments
        if len(x) == 1:
            y = globals()[current_f](x[0])
        else:
            y = globals()[current_f](*x)
        
        # Replace special characters in the result if it is a string
        if isinstance(y, str) and len(y) >= 1:
            y.replace('\\', '\\\\').replace('"', '\\"')
    except AssertionError:
        return float("inf"),
    except TypeError as e:
        # print(f"x is {x}")
        raise e
    
    # Get the total number of branches for the current function
    my_branches = branches[current_f.split("_instrumented")[0]]
    
    # Initialize fitness value
    fitness = 0.0
    
    # Sum up normalized branch distances 
    for branch in range(1, my_branches + 1):
        # for true branches
        if branch in distances_true:
            if distances_true[branch] == 0 and branch not in archive_true_branches:
                archive_true_branches[branch] = (x, y)
            if branch not in archive_true_branches:
                fitness += normalize(distances_true[branch])
        
        # for false branches
        if branch in distances_false:
            if distances_false[branch] == 0 and branch not in archive_false_branches:
                archive_false_branches[branch] = (x, y)
            if branch not in archive_false_branches:
                fitness += normalize(distances_false[branch])
    # print(f"my individual {x} fitness {fitness}")
    return fitness,

# Mutation function
def mut(elem):
    # print(elem[0])
    # individual is a tuple
    individual = elem[0]
    if len(individual) == 1:
        individual = (mutation(individual[0]), )
    else:
        individual = mutation(individual)
    
    elem[0] = individual
    return elem,

# Crossover function
def cross(elem1, elem2):
    parent1, parent2 = elem1[0], elem2[0]
    
    child1, child2 = crossover(parent1, parent2)
    
    elem1[0] = child1
    elem2[0] = child2
    return elem1, elem2

# Compile and Execute the code for Deap
def execute_function_deap(code):
    current_module = sys.modules[__name__]
    code = compile(code, filename="<string>", mode="exec")
    exec(code, current_module.__dict__)

# main function for the GA
def GA_deap():
    # Access global variables
    global archive_true_branches, archive_false_branches
    global list_true_archive, list_false_archive
    global current_f
    global file_path
    
    # create the toolbox
    toolbox = base.Toolbox()
    # create individual
    toolbox.register("attr_str", create_individual)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_str, n=1)
    
    # Set the population as a list of individuals
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Define the fitness evaluation function
    toolbox.register("evaluate", get_fitness)
    
    # Define the crossover operator
    toolbox.register("mate", cross)
    
    # Define the mutation operator
    toolbox.register("mutate", mut)
    
    
    # Define the selection operator
    toolbox.register("select", tools.selTournament, tournsize=TOURNSIZE)
    
    # Initialize an empty list to store coverage values
    coverage = []
    
    # Run the genetic algorithm for a specified number of repetitions (REPS)
    for i in range(REPS):
        # Reset archive dictionaries for true and false branches
        archive_true_branches = {}
        archive_false_branches = {}
        
        # Initialize a population of individuals
        population = toolbox.population(n=NPOP)
        
        # Run the Deap simple evolutionary algorithm
        algorithms.eaSimple(population, toolbox, CXPROB, MUPROB, NGEN, verbose=False)
        
        # Calculate the total coverage (sum of true and false branches)
        cov = len(archive_true_branches) + len(archive_false_branches)
        
        # Print coverage information for the current run
        # print(f"convered branches {cov} out of {my_branches * 2}")
        # print(f"true branches {archive_true_branches}")
        # print(f"false branches {archive_false_branches}\n")
        
        # Append the coverage value to the list
        coverage.append(cov)
    
    # Add information about the file, function, and covered branches to the global lists
    list_true_archive.append((file_path, current_f, dict(sorted(archive_true_branches.items()))))
    list_false_archive.append((file_path, current_f, dict(sorted(archive_false_branches.items()))))

# ------------------------ DEAP ------------------------ #

# for each file, run the GA and write the test in a "*file_name*_test.py", and copy and paste the folder Deap in the Archive folder
for i in range(1, COPIES + 1):
    print(f"copy {i}")
    list_true_archive, list_false_archive = [], []
    # loop over the files
    for instrumented_file in dict_codes:
        # get the name of the file
        file_path = instrumented_file
        # get the code of the file
        code = dict_codes[instrumented_file]
        # compile and execute the code
        execute_function_deap(code)
        
        # get the function names and the type of the parameters
        function_names = def_dict[instrumented_file][::2]
        parameters_type = def_dict[instrumented_file][1::2]
        
        # print(f"function names {function_names} and signatures {parameters_type}")
        
        # create the archive for the true, false branches and the coverage
        true_archive, false_archive, coverage_archive = [], [], []
        
        # for each function, run the GA and get the branches
        for current_f in function_names:
            current_arg = parameters_type[function_names.index(current_f)]
            branches = inst.get_branches_dict()
            # print(f"current function {current_f} with branches {branches}")
            my_branches = branches[current_f.split("_instrumented")[0]]
            GA_deap()
    
    # temp dict to create the test for true and false branches
    temp_dict_true = {}
    for elem in list_true_archive:
        key = elem[0]
        if key not in temp_dict_true:
            temp_dict_true[key] = []
        temp_dict_true[key].append((elem[1], elem[2]))
    
    temp_dict_false = {}
    for elem in list_false_archive:
        key = elem[0]
        if key not in temp_dict_false:
            temp_dict_false[key] = []
        temp_dict_false[key].append((elem[1], elem[2]))

    # write the test in a "*file_name*_test.py"
    write_test(temp_dict_true, val="true", folder = "Deap")
    write_test(temp_dict_false, val="false", folder = "Deap")
    
    # copy and paste the folder Deap in the Archive folder
    if not os.path.exists('Archive/fuzzer_test_archive/tests_fuzzer_copy_' + str(i)):
        original_folder = 'Deap'
        destination_folder = 'Archive/deap_test_archive/tests_deap_copy_' + str(i)
        shutil.copytree(original_folder, destination_folder)


# ----------------------- STATISTICAL COMPARISON ------------------------#

# get the json data from the mutation_scores.txt file
def get_json_data(file_path):
    
    """
    Parameters:
    - file_path: Path to the file containing the JSON data.
    
    Returns:
    - JSON data.
    """
    
    # Open the file and load the JSON data
    with open(file_path, 'r') as file:
        json_data = json.load(file)

    # return the json data
    return json_data


# Cohen's d is a statistical measure used to quantify the difference between two groups
# around 0.2 suggests that the difference between the groups is relatively small and may not have practical significance.
# around 0.5 suggests a moderate difference that might be of practical importance.
# around 0.8 or higher) indicates a substantial and potentially important difference between the groups.
# Compute Cohen's d effect size
def cohen_d(group1, group2):
    
    """
    Parameters:
    - group1: Array or list representing the first group of data.
    - group2: Array or list representing the second group of data.

    Returns:
    - Cohen's d effect size.
    """
    
    mean_diff = np.mean(group1) - np.mean(group2)
    pooled_std = np.sqrt((np.std(group1, ddof=1) ** 2 + np.std(group2, ddof=1) ** 2) / 2)

    effect_size = mean_diff / pooled_std

    return effect_size

# Plot boxplots for the Fuzzer and Deap scores
def plot_boxplots(df, name, fuzzer_avg, deap_avg, cohen_d_value, p_value, wilcoxon_result):
    
    """
    Parameters:
    - df: Dataframe containing the data to be plotted.
    - name: Name of the file being plotted.
    - fuzzer_avg: Average score for the Fuzzer.
    - deap_avg: Average score for the Deap.
    - cohen_d_value: Cohen's d effect size.
    - p_value: p-value of the Wilcoxon statistical test.
    - wilcoxon_result: Result of the Wilcoxon statistical test.
    
    Returns:
    - None
    """
    
    fig, ax = plt.subplots(figsize=(10, 6))

    # Boxplot for Fuzzer and DEAP Scores
    sns.boxplot(ax=ax, x='File_Name', y='Score', hue='Metric', data=df)

    # Set labels and title
    ax.set_xlabel('File Name')
    ax.set_ylabel('Mutation Score')
    ax.set_title(f'Mutation Score of {name}\n Fuzzy Avg: {round(fuzzer_avg, 3)}\n Deap Avg: {round(deap_avg, 3)}\n Cohen\'s d: {round(cohen_d_value, 3)}\n p-value: {p_value}: {wilcoxon_result}')

    # Adjust layout for better spacing
    plt.tight_layout()

    # Save the plot
    fig.savefig(f"Boxplots/{name}.png")


# for each file, create the dataframe for plotting purpose, calculate the cohen_d='s value and the Wilcoxon statistical test for the Fuzzer and Deap scores
def compute_statistical_comparison(json_data):
    
    """
    Parameters:
    - json_data: JSON data containing the scores for each file.
    
    Returns:
    - None
    """
    
    for file, scores in json_data.items():
        name = file.split('.')[0]
        fuzzer_scores = scores[0]
        deap_scores = scores[1]
        
        # Combine data for both Fuzzer and Deap scores into a single list
        combined_data = []
        combined_data.extend([(name, score, 'Fuzzer') for score in fuzzer_scores])
        combined_data.extend([(name, score, 'Deap') for score in deap_scores])
        df = pd.DataFrame(combined_data, columns=['File_Name', 'Score', 'Metric'])
        
        print(name, fuzzer_scores, deap_scores)
        
        # compute cohen's d
        cohen_d_value = cohen_d(fuzzer_scores, deap_scores)
        
        # compute Wilcoxon, zsplit is used to avoid zero-differences
        # it is a non-parametric statistical test used to determine if there is a significant difference between paired or matched observations.
        statistic, p_value = wilcoxon(fuzzer_scores, deap_scores, zero_method='zsplit')
        alpha = 0.05
        
        if p_value < alpha:
            wilcoxon_result = 'There is a significant difference'
        else:
            wilcoxon_result = 'No significant difference'
        
        plot_boxplots(df, name, np.mean(fuzzer_scores), np.mean(deap_scores), cohen_d_value, round(p_value, 3), wilcoxon_result)



file_path = f'mutation_scores.txt'
json_data = get_json_data(file_path)
compute_statistical_comparison(json_data)