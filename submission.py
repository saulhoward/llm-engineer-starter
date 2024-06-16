import argparse
import sys

from src.record import extract_from_pdf


def main(filepath: str):
    """Write the entrypoint to your submission here"""
    result = extract_from_pdf(filepath)
    sys.stdout.write(f"{result.model_dump_json()}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path-to-case-pdf",
        metavar="path",
        type=str,
        help="Path to local test case with which to run your code",
    )
    args = parser.parse_args()
    (
        main(args.path_to_case_pdf)
        if args.path_to_case_pdf
        else print("Please provide a PDF path")
    )
