"""A tool for providing final answers to user queries.

This module contains the FinalAnswerTool, which allows the LLM to provide a
final, well-formatted response to the user's query and signal the end of the
conversation.
"""

import json
from typing import Any


class FinalAnswerTool:
    """A special tool that allows the LLM to provide a final answer to the user's query.

    This tool should be used when the LLM has completed its task and wants to provide
    a clear, well-formatted final response to the user. Using this tool will signal
    that the conversation is complete and no further tool calls are needed.
    """

    async def run(
        self,
        answer: Any,
        answer_format: str = "text",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Process and store the final answer.

        Args:
            answer: The final answer to provide to the user. This can be any JSON-serializable value.
            answer_format: The format of the answer. Common values: "text", "markdown", "json", "html"
            metadata: Optional metadata to include with the answer, such as sources,
                    confidence scores, or other relevant information.

        Returns:
            A dictionary containing the answer, format, and metadata.
        """
        response = {
            "answer": answer,
            "format": answer_format,
            "metadata": metadata or {},
        }
        # Print the response
        print(f"{json.dumps(response)}")

        return response
