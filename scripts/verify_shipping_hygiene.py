#!/usr/bin/env python3
"""
GhostLayer Shipping Hygiene Verification Gate.

Guarantees that no release, archive, or PR can be exported with missing imports,
unbound typing symbols, broken module initializations, or pytest collection failures.

Runs 3 sequential verification layers:
1. Static AST Scan: Inspects all .py files for unbound typing symbols (List, Tuple, Dict, etc.).
2. Module Import Verification: Verifies that every module in ghost_layer imports cleanly.
3. Test Suite Collection Gate: Executes `pytest --collect-only` across all test files.

Returns exit code 0 if all checks pass cleanly, non-zero otherwise.
"""

import os
import sys
import ast
import subprocess
import importlib

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TYPING_NAMES = {"List", "Tuple", "Dict", "Optional", "Any", "Union", "Callable", "Set"}


def check_typing_annotations(target_dirs):
    """Layer 1: Static AST scan for unbound typing annotations."""
    print("========================================================")
    print("  LAYER 1: Static AST Analysis for Typing Hygiene")
    print("========================================================")
    
    violations = []
    total_files_scanned = 0

    for d in target_dirs:
        full_dir = os.path.join(ROOT_DIR, d)
        if not os.path.exists(full_dir):
            continue
        for root, _, files in os.walk(full_dir):
            for f in files:
                if not f.endswith(".py"):
                    continue
                file_path = os.path.join(root, f)
                total_files_scanned += 1
                rel_path = os.path.relpath(file_path, ROOT_DIR)
                
                try:
                    with open(file_path, "r", encoding="utf-8") as src:
                        tree = ast.parse(src.read(), filename=file_path)
                except Exception as e:
                    violations.append(f"{rel_path}: AST Parse Error: {e}")
                    continue

                imported_from_typing = set()
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.module == "typing":
                        for alias in node.names:
                            imported_from_typing.add(alias.name)
                    elif isinstance(node, ast.Import):
                        for alias in node.names:
                            if alias.name == "typing":
                                imported_from_typing.add("typing")

                # Check variable annotations
                for node in ast.walk(tree):
                    if isinstance(node, ast.AnnAssign):
                        for sub in ast.walk(node.annotation):
                            if isinstance(sub, ast.Name) and sub.id in TYPING_NAMES:
                                if sub.id not in imported_from_typing:
                                    violations.append(f"{rel_path}:{sub.lineno} - Unimported typing name '{sub.id}' in variable annotation")
                    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if node.returns:
                            for sub in ast.walk(node.returns):
                                if isinstance(sub, ast.Name) and sub.id in TYPING_NAMES:
                                    if sub.id not in imported_from_typing:
                                        violations.append(f"{rel_path}:{sub.lineno} - Unimported typing name '{sub.id}' in return annotation of '{node.name}'")
                        for arg in node.args.args + node.args.kwonlyargs:
                            if arg.annotation:
                                for sub in ast.walk(arg.annotation):
                                    if isinstance(sub, ast.Name) and sub.id in TYPING_NAMES:
                                        if sub.id not in imported_from_typing:
                                            violations.append(f"{rel_path}:{arg.lineno} - Unimported typing name '{sub.id}' in parameter '{arg.arg}'")

    print(f"Scanned {total_files_scanned} Python files across {target_dirs}.")
    if violations:
        print(f"[FAIL] Found {len(violations)} typing annotation violation(s):")
        for v in violations:
            print(f"  - {v}")
        return False
    else:
        print("[PASS] Zero unimported typing annotations detected.\n")
        return True


def check_module_imports():
    """Layer 2: Ensure ghost_layer and all submodules import cleanly."""
    print("========================================================")
    print("  LAYER 2: Runtime Module Import Verification")
    print("========================================================")
    
    ghost_dir = os.path.join(ROOT_DIR, "ghost_layer")
    failed_imports = []
    total_modules = 0

    # Ensure ROOT_DIR is on sys.path
    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)

    for root, _, files in os.walk(ghost_dir):
        for f in files:
            if not f.endswith(".py"):
                continue
            rel_path = os.path.relpath(os.path.join(root, f), ROOT_DIR)
            mod_name = rel_path[:-3].replace(os.sep, ".")
            total_modules += 1
            try:
                importlib.import_module(mod_name)
            except Exception as e:
                failed_imports.append((mod_name, f"{type(e).__name__}: {e}"))

    print(f"Tested import of {total_modules} ghost_layer modules.")
    if failed_imports:
        print(f"[FAIL] {len(failed_imports)} module(s) failed to import:")
        for mod, err in failed_imports:
            print(f"  - {mod}: {err}")
        return False
    else:
        print("[PASS] All modules imported cleanly.\n")
        return True


def check_pytest_collection():
    """Layer 3: Execute pytest --collect-only across the test suite."""
    print("========================================================")
    print("  LAYER 3: Pytest Collection Gate")
    print("========================================================")
    
    cmd = [sys.executable, "-m", "pytest", "--collect-only", "-q"]
    try:
        result = subprocess.run(cmd, cwd=ROOT_DIR, capture_output=True, text=True)
        if result.returncode != 0:
            print("[FAIL] Pytest collection failed:")
            print(result.stderr or result.stdout)
            return False
        
        # Count collected items from output
        lines = result.stdout.strip().splitlines()
        summary_line = lines[-1] if lines else "No output"
        print(f"[PASS] Pytest collection successful: {summary_line}\n")
        return True
    except Exception as e:
        print(f"[FAIL] Error running pytest collection: {e}")
        return False


def main():
    print("\nStarting GhostLayer Shipping Hygiene Verification Gate...\n")
    target_dirs = ["ghost_layer", "tests", "ceo_evidence_engine", "scripts"]
    
    l1_ok = check_typing_annotations(target_dirs)
    l2_ok = check_module_imports()
    l3_ok = check_pytest_collection()

    print("========================================================")
    print("  VERIFICATION SUMMARY")
    print("========================================================")
    print(f"  Layer 1 (AST Typing Scan)   : {'PASS' if l1_ok else 'FAIL'}")
    print(f"  Layer 2 (Module Imports)    : {'PASS' if l2_ok else 'FAIL'}")
    print(f"  Layer 3 (Pytest Collection) : {'PASS' if l3_ok else 'FAIL'}")
    print("--------------------------------------------------------")

    if l1_ok and l2_ok and l3_ok:
        print("RESULT: ALL SHIPPING HYGIENE GATES PASSED (SAFE TO SHIP)\n")
        sys.exit(0)
    else:
        print("RESULT: SHIPPING HYGIENE VIOLATIONS DETECTED (BLOCK SHIP)\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
