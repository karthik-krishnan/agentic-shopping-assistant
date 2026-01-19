from crewai import Crew, Process
from agent import product_expert, research_agent, manager_agent
from task import product_task, research_task, synthesis_task
from fallback import execute_with_fallback, log_task_completion, log_agent_step, FallbackResponse
from dotenv import load_dotenv

load_dotenv()


def create_crew():
    """Create and configure the hierarchical crew."""
    crew = Crew(
        agents=[product_expert, research_agent],
        tasks=[product_task, research_task, synthesis_task],
        process=Process.hierarchical,
        manager_agent=manager_agent,
        verbose=True,
        memory=False,  # Disabled: requires OPENAI_API_KEY for embeddings
        task_callback=log_task_completion,
        step_callback=log_agent_step
    )
    return crew


def main():
    """Main entry point with orchestration and error handling."""
    print("\n" + "=" * 60)
    print("CrewAI Multi-Agent Demo - Hierarchical Orchestration")
    print("=" * 60)
    print("\nAgents:")
    print("  - Product Expert: Recommends products based on specifications")
    print("  - Research Agent: Analyzes customer reviews and feedback")
    print("  - Manager Agent: Coordinates tasks and ensures quality")
    print("\nProcess: Hierarchical (Manager coordinates worker agents)")
    print("=" * 60 + "\n")

    # Create the crew
    crew = create_crew()

    # Execute with fallback logic
    result = execute_with_fallback(
        crew=crew,
        max_retries=2,
        fallback_type="product_search"
    )

    # Handle the result
    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)

    if isinstance(result, FallbackResponse):
        print(f"\nStatus: FALLBACK ACTIVATED")
        print(f"Message: {result.message}")
        if result.data:
            print(f"Additional info: {result.data}")
    else:
        print(f"\nStatus: SUCCESS")
        print(f"\n{result}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
