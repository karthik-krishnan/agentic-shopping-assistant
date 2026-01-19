import time
from typing import Any, Optional
from dataclasses import dataclass


@dataclass
class FallbackResponse:
    """Structure for fallback responses when agents fail."""
    success: bool
    message: str
    data: Optional[Any] = None


# Default fallback responses for common scenarios
FALLBACK_RESPONSES = {
    "product_search": FallbackResponse(
        success=False,
        message="We're experiencing technical difficulties. Please try again later.",
        data={
            "suggestion": "In the meantime, you can browse our catalog at /products",
            "contact": "For immediate assistance, contact support@example.com"
        }
    ),
    "research": FallbackResponse(
        success=False,
        message="Unable to fetch product reviews at this time.",
        data={
            "suggestion": "Check trusted review sites like Consumer Reports or RunningShoeGuru",
            "note": "Our team is working to restore this feature"
        }
    ),
    "general": FallbackResponse(
        success=False,
        message="An unexpected error occurred. Our team has been notified.",
        data=None
    )
}


def execute_with_fallback(crew, inputs: dict = None, max_retries: int = 2,
                          fallback_type: str = "general") -> Any:
    """
    Execute a crew with retry logic and fallback responses.

    Args:
        crew: The CrewAI Crew instance to execute
        inputs: Optional inputs to pass to crew.kickoff()
        max_retries: Maximum number of retry attempts (default: 2)
        fallback_type: Type of fallback response to use on failure

    Returns:
        Crew result on success, FallbackResponse on failure
    """
    last_error = None

    for attempt in range(max_retries):
        try:
            print(f"\n{'='*50}")
            print(f"Execution attempt {attempt + 1} of {max_retries}")
            print(f"{'='*50}\n")

            if inputs:
                result = crew.kickoff(inputs=inputs)
            else:
                result = crew.kickoff()

            print(f"\n{'='*50}")
            print("Execution completed successfully!")
            print(f"{'='*50}\n")

            return result

        except Exception as e:
            last_error = e
            print(f"\n{'!'*50}")
            print(f"Attempt {attempt + 1} failed: {type(e).__name__}: {str(e)}")
            print(f"{'!'*50}\n")

            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s...
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)

    # All retries exhausted, return fallback response
    print(f"\n{'#'*50}")
    print("All retry attempts exhausted. Returning fallback response.")
    print(f"Last error: {last_error}")
    print(f"{'#'*50}\n")

    return get_fallback_response(fallback_type, error=last_error)


def get_fallback_response(fallback_type: str = "general",
                          error: Exception = None) -> FallbackResponse:
    """
    Get an appropriate fallback response based on the failure type.

    Args:
        fallback_type: Type of operation that failed
        error: The exception that caused the failure

    Returns:
        FallbackResponse with appropriate message and suggestions
    """
    response = FALLBACK_RESPONSES.get(fallback_type, FALLBACK_RESPONSES["general"])

    # Enhance response with error details if available
    if error and response.data:
        response.data["error_type"] = type(error).__name__

    return response


def log_task_completion(output) -> None:
    """Callback for logging task completions."""
    print(f"\n{'~'*50}")
    print("Task Completed")
    print(f"{'~'*50}")
    if hasattr(output, 'description'):
        print(f"Task: {output.description[:100]}...")
    if hasattr(output, 'raw'):
        preview = output.raw[:200] if len(output.raw) > 200 else output.raw
        print(f"Output preview: {preview}...")
    print(f"{'~'*50}\n")


def log_agent_step(step_output) -> None:
    """Callback for logging individual agent steps."""
    print(f"\n[Step] Agent action recorded")
    if hasattr(step_output, 'output'):
        preview = str(step_output.output)[:100]
        print(f"[Step] Output: {preview}...")
