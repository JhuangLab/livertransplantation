#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run the protein enrichment R script.

This script:
- Checks that the R script and consensus feature summary CSV exist
- Verifies that Rscript is available on PATH
- Executes the R script in the current directory
"""

import os
import subprocess
import sys
from pathlib import Path


def main():
    script_dir = Path(__file__).parent.absolute()
    r_script = script_dir / "protein_analysis.R"
    summary_csv = script_dir / "top50_features_two_comparisons" / "features_by_omics_and_comparison.csv"

    if not r_script.exists():
        print(f"Error: R script not found at {r_script}")
        sys.exit(1)

    if not summary_csv.exists():
        print(f"Error: required input file not found: {summary_csv}")
        print("Please make sure the consensus feature summary CSV exists.")
        sys.exit(1)

    original_dir = os.getcwd()
    try:
        os.chdir(script_dir)

        print("=" * 60)
        print("Running Protein Enrichment Analysis (R)")
        print("=" * 60)
        print(f"R script: {r_script}")
        print(f"Working directory: {script_dir}")
        print()

        # Check Rscript availability
        try:
            version_check = subprocess.run(
                ["Rscript", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if version_check.returncode != 0:
                print("Warning: Rscript may not be properly configured.")
        except FileNotFoundError:
            print("Error: Rscript not found. Please install R and ensure Rscript is on PATH.")
            sys.exit(1)
        except subprocess.TimeoutExpired:
            print("Warning: Rscript version check timed out.")

        print("Executing R script...")
        print("-" * 60)

        result = subprocess.run(
            ["Rscript", str(r_script)],
            check=False,
            capture_output=False,
            text=True,
        )

        if result.returncode == 0:
            print()
            print("-" * 60)
            print("Protein enrichment analysis completed successfully.")
            print(f"Results saved in: {script_dir / 'protein_enrichment_results'}")
        else:
            print()
            print("-" * 60)
            print(f"Error: R script exited with code {result.returncode}")
            sys.exit(result.returncode)

    except Exception as exc:
        print(f"Error executing R script: {exc}")
        sys.exit(1)
    finally:
        os.chdir(original_dir)


if __name__ == "__main__":
    main()