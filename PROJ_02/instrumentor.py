import ast
import os

from nltk.metrics.distance import edit_distance

# ------------------------ Dictionary ------------------------ #

# Dictionaries to store distances for true and false branches
distance_dict_true: dict[int, int] = {}
distance_dict_false: dict[int, int] = {}

# Dictionary to store branch numbers for each function
branches_dict: dict[str, int] = {}

def get_distance_dicts():
    return distance_dict_true, distance_dict_false

# ---------------------- Original filename ------------------ #

# Function to get the list of original file names
def get_original_filenames():
    return list_of_files

# Function to get the dictionary of branches for each function
def get_branches_dict():
    return branches_dict

# ------------------------ Transformer ------------------------ #

# Custom transformer class that modifies the AST
class Transformer(ast.NodeTransformer):
    branch_num = 0
    
    name_lists = []
    
    node_name = ""
    
    number_of_functions = 0
    number_of_comparisons = 0
    
    global branches_dict
    
    # FunctionDef node handler
    def visit_FunctionDef(self, node):
        self.branch_num = 0
        self.name_lists.append(node.name)
        self.node_name = node.name
        branches_dict[self.node_name] = self.branch_num
        
        node.name = node.name + "_instrumented"
        
        self.number_of_functions += 1
        
        return self.generic_visit(node)
    
    
    # Compare node handler
    def visit_Compare(self, node):
        if node.ops[0] in [ast.Is, ast.IsNot, ast.In, ast.NotIn]:
            return node
        
        self.branch_num += 1
        branches_dict[self.node_name] = self.branch_num
        
        self.number_of_comparisons += 1
        
        return ast.Call(func=ast.Name("evaluate_condition", ast.Load()),
                        args=[ast.Num(self.branch_num),
                              ast.Str(node.ops[0].__class__.__name__),
                              node.left,
                              node.comparators[0]],
                        keywords=[],
                        starargs=None,
                        kwargs=None)
    
    # Assert node handler
    def visit_Assert(self, node):
        return node
    
    # Call node handler
    def visit_Call(self, node):
        try:
            if node.func.id in self.name_lists:
                node.func.id = node.func.id + "_instrumented"
        except:
            pass
        
        return node
    
    # Return node handler
    def visit_Return(self, node):
        if isinstance(node.value, ast.Call):
            return self.generic_visit(node)
        else:
            return node

#  ------------------------ evaluate_condition ------------------------ #

# Function to evaluate conditions and update distance dictionaries
def evaluate_condition(num, op, lhs, rhs):
    
    distance_true = 0
    distance_false = 0
    
    long_l = -1
    long_r = -1
    
    # Convert single-character strings to their ASCII values
    if isinstance(lhs, str):
        if len(lhs) == 1:
            lhs = ord(lhs)
        else:
            long_l = len(lhs)
    
    # Convert single-character strings to their ASCII values
    if isinstance(rhs, str):
        if len(rhs) == 1:
            rhs = ord(rhs)
        else:
            long_r = len(rhs)
    
    # Calculate distances based on comparison operator
    if long_l != -1 or long_r != -1:
        # String comparison
        if op == "Eq":
            distance_true = edit_distance(lhs, rhs)
            distance_false = 1 if lhs == rhs else 0
        elif op == "NotEq":
            distance_true = 1 if lhs == rhs else 0
            distance_false = edit_distance(lhs, rhs)
    else:
        # Numeric comparison
        if op == "Lt":
            distance_true = lhs - rhs + 1 if lhs >= rhs else 0
            distance_false = rhs - lhs if lhs < rhs else 0
        elif op == "Gt":
            distance_true = rhs - lhs + 1 if lhs <= rhs else 0
            distance_false = lhs - rhs if lhs > rhs else 0
        elif op == "LtE":
            distance_true = lhs - rhs if lhs > rhs else 0
            distance_false = rhs - lhs + 1 if lhs <= rhs else 0
        elif op == "GtE":
            distance_true = rhs - lhs if lhs < rhs else 0
            distance_false = lhs - rhs + 1 if lhs >= rhs else 0
        elif op == "Eq":
            distance_true = abs(lhs - rhs) if lhs != rhs else 0
            distance_false = 1 if lhs == rhs else 0
        elif op == "NotEq":
            distance_true = 1 if lhs == rhs else 0
            distance_false = abs(lhs - rhs) if lhs != rhs else 0
        else:
            pass
    
    # Update distance dictionaries
    update_maps(num, distance_true, distance_false)
    return True if distance_true == 0 else False

# ------------------------ update_maps ------------------------ #

# Function to update distance dictionaries
def update_maps(condition_num, d_true, d_false):
    global distance_dict_true, distance_dict_false
    
    if condition_num in distance_dict_true:
        if distance_dict_true[condition_num] <= d_true:
            distance_dict_true[condition_num] = distance_dict_true[condition_num]
    else:
        distance_dict_true[condition_num] = d_true

    if condition_num in distance_dict_false:
        if distance_dict_false[condition_num] <= d_false:
            distance_dict_false[condition_num] = distance_dict_false[condition_num]
    else:
        distance_dict_false[condition_num] = d_false


# ------------------------ Main ------------------------ #

# Directory containing benchmark files
files_in_benchmark = "benchmark"
list_of_files = []

for filename in os.listdir(files_in_benchmark):
    f = os.path.join(files_in_benchmark, filename)
    # checking if it is a file
    if os.path.isfile(f):
        # print(f)
        list_of_files.append(f)
# [print(x) for x in list_of_files]


# Create an instance of the Transformer class
transformer = Transformer()

# Process each file in the benchmark directory
for x in list_of_files:
    
    test_code = x
    code = open(test_code, "r")
    f = code.read()
    # print(f)
    
    # Parse the code and apply the transformer
    tree = ast.parse(f)
    new_tree = transformer.visit(tree)
    
    # Unparse the modified tree
    new_parsed = ast.unparse(new_tree)
    
    # Extract the name of the file without the path and extension
    name = x.replace("benchmark/", "")
    name = name.replace(".py", "")
    # print(name)
    
    # Construct the path for the new instrumented file
    new_python_file = "parsed/" + name + "_instrumented.py"
    
    # Write the instrumented code to the new file
    with open(new_python_file, 'w') as file:
        file.write("from instrumentor import evaluate_condition\n\n\n")
        file.write(new_parsed)

# Print information for the report
print("Number of files in parsed folder: ", len(list_of_files))
print("Number of functions: ", transformer.number_of_functions)
print("Number of comparisons: ", transformer.number_of_comparisons)









