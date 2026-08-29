"""Project metadata extraction and inference."""

from local_coding_assistant.repository.metadata.detector import MetadataFileDetector
from local_coding_assistant.repository.metadata.extractor import MetadataExtractor
from local_coding_assistant.repository.metadata.framework_inference import (
    FrameworkInference,
)

__all__ = [
    "FrameworkInference",
    "MetadataExtractor",
    "MetadataFileDetector",
]
