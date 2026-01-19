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


product_task = Task(
    description="""Identify running shoes from our catalog that match these criteria:
    - Waterproof or water-resistant
    - Priced under $100

    Provide the product names, key features, and prices.""",
    expected_output="A list of waterproof running shoes under $100 with their features and prices.",
    agent=product_expert,
    guardrail=validate_product_response,
    guardrail_max_retries=2
)

research_task = Task(
    description="""Research customer reviews and feedback for waterproof running shoes.
    Focus on:
    - Common praise points (comfort, durability, waterproofing effectiveness)
    - Common complaints or issues
    - Overall customer satisfaction ratings

    Provide a summary of findings that would help a customer make a decision.""",
    expected_output="A summary of customer reviews highlighting pros, cons, and ratings for waterproof running shoes.",
    agent=research_agent,
    guardrail=validate_research_response,
    guardrail_max_retries=2
)

synthesis_task = Task(
    description="""Combine the product recommendations and research findings to provide
    a final recommendation. Consider:
    - Product features and specifications
    - Customer feedback and satisfaction
    - Value for money

    Provide a prioritized recommendation with clear reasoning.""",
    expected_output="A final recommendation combining product details with customer feedback, ranked by overall value.",
    agent=product_expert,
    context=[product_task, research_task]
)
