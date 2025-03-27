#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import pytest
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tests for FLA components")
    parser.add_argument(
        "--test-file", 
        default=None, 
        help="Specific test file to run (default: run all tests)"
    )
    parser.add_argument(
        "--verbose", 
        "-v", 
        action="store_true", 
        help="Enable verbose output"
    )
    parser.add_argument(
        "--precision", 
        choices=["float16", "bfloat16", "both"], 
        default="both",
        help="Precision to test with"
    )
    
    args = parser.parse_args()
    
    # Find the tests directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Default to running all tests
    if args.test_file is None:
        test_path = current_dir
    else:
        test_path = os.path.join(current_dir, args.test_file)
    
    # Build pytest arguments
    pytest_args = [test_path]
    
    # Add verbosity if requested
    if args.verbose:
        pytest_args.append("-v")
    
    # Filter tests by precision if specified
    if args.precision != "both":
        pytest_args.extend(["-k", args.precision])
    
    # Run the tests
    sys.exit(pytest.main(pytest_args)) 
