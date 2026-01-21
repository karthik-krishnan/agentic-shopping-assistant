import argparse
import logging
import sys
from datetime import datetime
from contextlib import contextmanager
from crewai import Crew, Process
from agent import product_expert, research_agent, manager_agent
from task import create_tasks
from fallback import execute_with_fallback, FallbackResponse
from dotenv import load_dotenv

load_dotenv()

# Parse command line arguments early
parser = argparse.ArgumentParser(description='CrewAI Multi-Agent Demo')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='Enable verbose output with colored agent/task logs')
args = parser.parse_args()

VERBOSE_MODE = args.verbose


@contextmanager
def redirect_stdout_to_file(filepath):
    """Redirect stdout/stderr to a file during execution."""
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with open(filepath, 'a') as f:
        sys.stdout = f
        sys.stderr = f
        try:
            yield
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


@contextmanager
def no_redirect():
    """No-op context manager for verbose mode."""
    yield


# Configure logging
LOG_FILE = f"crew_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Store original stdout before any redirection
_original_stdout = sys.stdout

# File handler for detailed logs (captures everything)
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

if not VERBOSE_MODE:
    # Redirect all CrewAI/LiteLLM logging to file only (concise mode)
    for logger_name in ['crewai', 'litellm', 'openai', 'httpx', 'httpcore', 'urllib3']:
        lib_logger = logging.getLogger(logger_name)
        lib_logger.handlers = []
        lib_logger.addHandler(file_handler)
        lib_logger.setLevel(logging.DEBUG)
        lib_logger.propagate = False

    # Root logger - send to file only to catch any other verbose output
    root_logger = logging.getLogger()
    root_logger.handlers = []
    root_logger.addHandler(file_handler)
    root_logger.setLevel(logging.DEBUG)

# Our app logger - shows concise output on console (uses original stdout)
console_handler = logging.StreamHandler(_original_stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(message)s'))

logger = logging.getLogger('app')
logger.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.propagate = False


def log_task_completion(output) -> None:
    """Callback for logging task completions."""
    task_desc = getattr(output, 'description', 'Unknown task')[:50]
    logger.info(f"  [DONE] {task_desc}...")
    if hasattr(output, 'raw'):
        logger.debug(f"Task output: {output.raw}")


def log_agent_step(step_output) -> None:
    """Callback for logging individual agent steps."""
    if hasattr(step_output, 'output'):
        logger.debug(f"Agent step: {step_output.output}")


def create_crew(user_query: str):
    """Create and configure the hierarchical crew based on user query."""
    product_task, research_task, synthesis_task = create_tasks(user_query)

    # Set agent verbosity based on mode
    product_expert.verbose = VERBOSE_MODE
    research_agent.verbose = VERBOSE_MODE
    manager_agent.verbose = VERBOSE_MODE

    crew = Crew(
        agents=[product_expert, research_agent],
        tasks=[product_task, research_task, synthesis_task],
        process=Process.hierarchical,
        manager_agent=manager_agent,
        verbose=VERBOSE_MODE,
        memory=False,  # Disabled: requires OPENAI_API_KEY for embeddings
        task_callback=log_task_completion,
        step_callback=log_agent_step
    )
    return crew


def main():
    """Main entry point with orchestration and error handling."""
    logger.info("CrewAI Multi-Agent Demo")
    mode_str = "verbose" if VERBOSE_MODE else "concise"
    logger.info(f"Mode: {mode_str} (use -v for verbose colored output)")
    logger.info("-" * 40)

    # Get user input
    user_query = input("What product are you looking for? ").strip()
    if not user_query:
        logger.info("No input provided. Exiting.")
        return

    logger.info(f"Searching for: {user_query}")
    if not VERBOSE_MODE:
        logger.info(f"Detailed logs: {LOG_FILE}")
    logger.info("-" * 40)

    # Create the crew with user's query
    crew = create_crew(user_query)

    # Execute with fallback logic
    logger.info("Starting crew execution...")
    redirect_ctx = no_redirect() if VERBOSE_MODE else redirect_stdout_to_file(LOG_FILE)
    with redirect_ctx:
        result = execute_with_fallback(
            crew=crew,
            max_retries=2,
            fallback_type="product_search",
            logger=logger
        )

    # Handle the result
    logger.info("-" * 40)
    if isinstance(result, FallbackResponse):
        logger.info(f"Status: FALLBACK - {result.message}")
        logger.debug(f"Fallback data: {result.data}")
    else:
        logger.info("Status: SUCCESS")
        logger.info(f"\n{result}")
        logger.debug(f"Full result: {result}")

    logger.info("-" * 40)


if __name__ == "__main__":
    main()
