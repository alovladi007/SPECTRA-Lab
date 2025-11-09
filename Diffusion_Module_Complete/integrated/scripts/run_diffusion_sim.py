#!/usr/bin/env python3
"""
CLI tool for batch diffusion simulations.

Usage:
    python scripts/run_diffusion_sim.py --input runs.csv --output results.parquet

Will be implemented in Session 10.
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Batch diffusion simulations")
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--output", required=True, help="Output file")
    args = parser.parse_args()
    
    raise NotImplementedError("Session 10: Batch diffusion simulations")


if __name__ == "__main__":
    main()
