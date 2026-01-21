from crewai import Task
from crewai.tasks.task_output import TaskOutput
from typing import Tuple, Any
from agent import product_expert, research_agent


def validate_product_response(result: TaskOutput) -> Tuple[bool, Any]:
    """Validate that the response contains actual product information."""
    output = result.raw.lower()

    # Check for unhelpful responses
    unhelpful_phrases = ["i don't know", "i cannot", "no products found", "unable to find"]
    if any(phrase in output for phrase in unhelpful_phrases):
        return (False, "Response lacks product details. Please try again with more specific information.")

    # Check for minimum content length
    if len(output) < 50:
        return (False, "Response too brief. Please provide more detailed information.")

    return (True, result)


def validate_research_response(result: TaskOutput) -> Tuple[bool, Any]:
    """Validate that research contains meaningful insights."""
    output = result.raw.lower()

    # Check for actual research content
    if len(output) < 100:
        return (False, "Research findings too brief. Please provide more comprehensive analysis.")

    return (True, result)


def create_tasks(user_query: str):
    """Create tasks based on user's product search query."""
    product_task = Task(
        description=f"""Identify products from our catalog that match the user's request:
        "{user_query}"

        Provide the product names, key features, and prices.""",
        expected_output=f"A list of products matching '{user_query}' with their features and prices.",
        agent=product_expert,
        guardrail=validate_product_response,
        guardrail_max_retries=2
    )

    research_task = Task(
        description=f"""Research customer reviews and feedback for products matching:
        "{user_query}"

        Focus on:
        - Common praise points (quality, durability, value)
        - Common complaints or issues
        - Overall customer satisfaction ratings

        Provide a summary of findings that would help a customer make a decision.""",
        expected_output=f"A summary of customer reviews highlighting pros, cons, and ratings for '{user_query}'.",
        agent=research_agent,
        guardrail=validate_research_response,
        guardrail_max_retries=2
    )

    synthesis_task = Task(
        description=f"""Combine the product recommendations and research findings to provide
        a final recommendation for the user's request: "{user_query}"

        Consider:
        - Product features and specifications
        - Customer feedback and satisfaction
        - Value for money

        Provide a prioritized recommendation with clear reasoning.""",
        expected_output="A final recommendation combining product details with customer feedback, ranked by overall value.",
        agent=product_expert,
        context=[product_task, research_task]
    )

    return product_task, research_task, synthesis_task
