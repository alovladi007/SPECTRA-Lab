#!/usr/bin/env python3
"""
CLI tool for batch oxidation simulations.

Will be implemented in Session 10.
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Batch oxidation simulations")
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--output", required=True, help="Output file")
    args = parser.parse_args()
    
    raise NotImplementedError("Session 10: Batch oxidation simulations")


if __name__ == "__main__":
    main()
