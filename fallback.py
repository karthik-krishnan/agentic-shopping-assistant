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
                          fallback_type: str = "general", logger=None) -> Any:
    """
    Execute a crew with retry logic and fallback responses.

    Args:
        crew: The CrewAI Crew instance to execute
        inputs: Optional inputs to pass to crew.kickoff()
        max_retries: Maximum number of retry attempts (default: 2)
        fallback_type: Type of fallback response to use on failure
        logger: Optional logger for output

    Returns:
        Crew result on success, FallbackResponse on failure
    """
    def log(msg, level="info"):
        if logger:
            getattr(logger, level)(msg)
        else:
            print(msg)

    last_error = None

    for attempt in range(max_retries):
        try:
            log(f"  Attempt {attempt + 1}/{max_retries}...")

            if inputs:
                result = crew.kickoff(inputs=inputs)
            else:
                result = crew.kickoff()

            log("  Execution completed!")
            return result

        except Exception as e:
            last_error = e
            log(f"  Attempt {attempt + 1} failed: {type(e).__name__}", "warning")
            log(f"  Error details: {str(e)}", "debug")

            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                log(f"  Retrying in {wait_time}s...")
                time.sleep(wait_time)

    log("All retries exhausted. Using fallback.", "warning")
    log(f"Last error: {last_error}", "debug")

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


