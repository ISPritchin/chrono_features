#!/usr/bin/env python3
"""Script to automatically add parameter names to function calls in Python code.
"""

import ast
import os
import sys
import argparse
from typing import List


class FunctionDefinitionVisitor(ast.NodeVisitor):
    """AST visitor to collect function and method definitions."""
    
    def __init__(self):
        self.functions = {}  # name -> {params: [param_names], lineno: line_number}
        self.methods = {}    # class_name.method_name -> {params: [param_names], lineno: line_number}
        self.current_class = None
    
    def visit_ClassDef(self, node):
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class
    
    def visit_FunctionDef(self, node):
        param_names = []
        for arg in node.args.args:
            if arg.arg != "self" and arg.arg != "cls":
                param_names.append(arg.arg)
        
        if self.current_class:
            key = f"{self.current_class}.{node.name}"
            self.methods[key] = {
                "params": param_names,
                "lineno": node.lineno,
            }
        else:
            self.functions[node.name] = {
                "params": param_names,
                "lineno": node.lineno,
            }
        
        self.generic_visit(node)


class FunctionCallTransformer(ast.NodeTransformer):
    """AST transformer to add parameter names to function calls."""
    
    def __init__(self, functions, methods):
        self.functions = functions
        self.methods = methods
        self.current_class = None
        self.imported_names = set()
        self.modified = False
        self.class_instances = {}  # Track class instances: variable_name -> class_name
    
    def visit_ImportFrom(self, node):
        for name in node.names:
            if name.asname:
                self.imported_names.add(name.asname)
            else:
                self.imported_names.add(name.name)
        return node
    
    def visit_Import(self, node):
        for name in node.names:
            if name.asname:
                self.imported_names.add(name.asname)
            else:
                module_name = name.name.split(".")[0]
                self.imported_names.add(module_name)
        return node
    
    def visit_ClassDef(self, node):
        old_class = self.current_class
        self.current_class = node.name
        node = self.generic_visit(node)
        self.current_class = old_class
        return node
    
    def visit_Assign(self, node):
        """Track class instantiations to identify instance types."""
        node = self.generic_visit(node)
        
        # Check if this is a class instantiation
        if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and 
            isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)):
            var_name = node.targets[0].id
            class_name = node.value.func.id
            
            # Only track if we know this is a class defined in our code
            if class_name in [cls_name.split(".")[0] for cls_name in self.methods.keys()]:
                self.class_instances[var_name] = class_name
        
        return node
    
    def visit_Call(self, node):
        # Process any nested calls first
        node = self.generic_visit(node)
        
        # Skip if already has keyword arguments or if it's a built-in/imported function
        if node.keywords:
            return node
        
        func_name = None
        method_name = None
        
        # Handle direct function calls
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in self.imported_names or func_name.startswith("_"):
                return node
        
        # Handle method calls
        elif isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                if self.current_class and node.func.value.id == "self":
                    # Method call on self within a class
                    method_name = f"{self.current_class}.{node.func.attr}"
                elif node.func.value.id in self.class_instances:
                    # Method call on a known class instance
                    class_name = self.class_instances[node.func.value.id]
                    method_name = f"{class_name}.{node.func.attr}"
            else:
                # Skip other method calls
                return node
        else:
            return node
        
        # Get parameter names
        param_names = []
        if func_name and func_name in self.functions:
            param_names = self.functions[func_name]["params"]
        elif method_name and method_name in self.methods:
            param_names = self.methods[method_name]["params"]
        
        # Add parameter names to positional arguments
        if param_names and len(node.args) <= len(param_names):
            new_keywords = []
            for i, arg in enumerate(node.args):
                new_keywords.append(ast.keyword(arg=param_names[i], value=arg))
            
            if new_keywords:
                self.modified = True
                node.keywords = new_keywords
                node.args = []
        
        return node


def process_file(file_path: str, dry_run: bool = False) -> bool:
    """Process a single Python file to add parameter names to function calls.
    
    Args:
        file_path: Path to the Python file to process
        dry_run: If True, don't modify the file, just print what would be changed
        
    Returns:
        True if the file was modified, False otherwise
    """
    with open(file_path, encoding="utf-8") as f:
        content = f.read()
    
    # Parse the file
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        print(f"Syntax error in {file_path}: {e}")
        return False
    
    # First pass: collect function and method definitions
    visitor = FunctionDefinitionVisitor()
    visitor.visit(tree)
    
    # Second pass: transform function calls
    transformer = FunctionCallTransformer(visitor.functions, visitor.methods)
    new_tree = transformer.visit(tree)
    
    if transformer.modified:
        # Generate the modified code
        new_content = ast.unparse(new_tree)
        
        if dry_run:
            print(f"Would modify {file_path}")
            print("=" * 80)
            print(new_content)
            print("=" * 80)
        else:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            print(f"Modified {file_path}")
        
        return True
    
    return False


def find_python_files(directory: str) -> List[str]:
    """Find all Python files in a directory and its subdirectories.
    
    Args:
        directory: Directory to search for Python files
        
    Returns:
        List of paths to Python files
    """
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
    return python_files


def main():
    parser = argparse.ArgumentParser(description="Add parameter names to function calls in Python code")
    parser.add_argument("path", help="Path to a Python file or directory")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without modifying files")
    parser.add_argument("--recursive", "-r", action="store_true", help="Process directories recursively")
    
    args = parser.parse_args()
    
    if os.path.isfile(args.path) and args.path.endswith(".py"):
        process_file(args.path, args.dry_run)
    elif os.path.isdir(args.path):
        if args.recursive:
            files = find_python_files(args.path)
        else:
            files = [os.path.join(args.path, f) for f in os.listdir(args.path) 
                    if os.path.isfile(os.path.join(args.path, f)) and f.endswith(".py")]
        
        modified_count = 0
        for file_path in files:
            if process_file(file_path, args.dry_run):
                modified_count += 1
        
        print(f"Modified {modified_count} out of {len(files)} files")
    else:
        print(f"Error: {args.path} is not a Python file or directory")
        sys.exit(1)


if __name__ == "__main__":
    main()