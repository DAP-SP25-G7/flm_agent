"""CLI script to run the full ingestion pipeline."""

import argparse
import sys

from loguru import logger

from src.ingestion.pipeline import run_ingestion


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the FLM Agent ingestion pipeline.")
    parser.add_argument(
        "--skip-embed",
        action="store_true",
        help="Skip the embedding/Pinecone upsert step.",
    )
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    summary = run_ingestion(skip_embed=args.skip_embed)
    print(f"\nPipeline summary: {summary}")


if __name__ == "__main__":
    main()
