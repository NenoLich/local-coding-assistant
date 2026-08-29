"""Example Python file for testing the repository context pipeline."""

import asyncio
from typing import List


class DataProcessor:
    """A class for processing data."""

    def __init__(self, batch_size: int = 100):
        self.batch_size = batch_size
        self.processed_count = 0

    async def process_batch(self, data: List[str]) -> dict:
        """Process a batch of data asynchronously.

        Args:
            data: List of strings to process.

        Returns:
            Dictionary with processing results.
        """
        results = {"count": len(data), "status": "success"}
        self.processed_count += len(data)
        return results

    def validate_input(self, value: str) -> bool:
        """Validate input string.

        Args:
            value: String to validate.

        Returns:
            True if valid, False otherwise.
        """
        return bool(value and value.strip())


def calculate_metrics(data: List[dict]) -> dict:
    """Calculate metrics from processed data.

    Args:
        data: List of processed data dictionaries.

    Returns:
        Dictionary with calculated metrics.
    """
    if not data:
        return {"total": 0, "average": 0.0}

    total = sum(d.get("count", 0) for d in data)
    average = total / len(data)

    return {"total": total, "average": average}


async def main():
    """Main entry point for data processing."""
    processor = DataProcessor(batch_size=50)
    sample_data = ["item1", "item2", "item3"]

    result = await processor.process_batch(sample_data)
    metrics = calculate_metrics([result])

    print(f"Processed: {metrics}")


if __name__ == "__main__":
    asyncio.run(main())
