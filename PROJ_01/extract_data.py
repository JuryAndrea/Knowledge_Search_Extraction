import os
import pandas as pd
import ast

data_list = []


class Kse_visitor(ast.NodeVisitor):

    path = ""

    # check if the name starts with "_" or "main" or contains "test"
    def isblacklist(self, node):
        name = node.name
        if name.startswith("_") or name == "main" or "test" in name.lower():
            return True
        return False

    # visit and save the path
    def visit_path(self, path):
        self.path = path
        self.visit(ast.parse(open(path).read()))

    # add name, path, line, node_type and commet to a list as a tuple
    def add_data(self, node, node_type):
        name = node.name
        line = node.lineno
        path = self.path
        str = ast.get_docstring(node)
        # force the comment to be on a single line
        if str == None:
            comment = " "
        else:
            comment = "".join(line.strip() for line in str.splitlines())
        
        data_list.append([name, path, line, node_type, comment])

    # Override visit_FunctionDef method
    def visit_FunctionDef(self, node: ast.FunctionDef):
        if self.isblacklist(node):
            return
        # print(node.name, "Function")
        self.add_data(node, "Function")

    # Overide visit_ClassDef method
    def visit_ClassDef(self, node: ast.ClassDef):
        if self.isblacklist(node):
            return
        # print(node.name, "Class")
        self.add_data(node, "Class")
        # visit all the children nodes
        for child_node in node.body:
            if isinstance(child_node, ast.FunctionDef):
                if not self.isblacklist(child_node):
                    # print(node.name, "Method")
                    self.add_data(child_node, "Method")


paths_list = []
# starting folder
path = "tensorflow/"
# searching for all python files and append them to a list
for (root, dirs, files) in os.walk(path, topdown=True):
    for name in files:
        if name.endswith(".py"):
            paths_list.append(os.path.join(root, name))

print("number of python files: ", len(paths_list))
# print(paths_list)

# visit all the paths
visitor = Kse_visitor()
for path in paths_list:
    visitor.visit_path(path)

# create a dataframe and count Classes, Functions and Methods
df = pd.DataFrame(data_list, columns=[
                  "Name", "Path", "Line", "Type", "Comment"])
# print(df)
df.to_csv("data.csv", index=False)
print("class: ", df['Type'].value_counts()['Class'])
print("function: ", df['Type'].value_counts()['Function'])
print("method: ", df['Type'].value_counts()['Method'])
