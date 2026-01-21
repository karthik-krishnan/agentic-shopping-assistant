from crewai import Agent, LLM
import os
from dotenv import load_dotenv

load_dotenv()

llm = LLM(
    model="azure/" + os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0.7
)

product_expert = Agent(
    role="Product Expert",
    goal="Help users understand products and suggest options based on features and specifications",
    backstory="""You are a retail domain expert specializing in performance footwear.
    You have deep knowledge of product features, materials, and technical specifications.
    You provide accurate product recommendations based on customer needs.""",
    llm=llm,
    verbose=False,
    allow_delegation=True
)

research_agent = Agent(
    role="Product Researcher",
    goal="Find and analyze product reviews, ratings, and customer feedback",
    backstory="""You are an expert researcher who excels at finding product reviews
    and customer feedback. You analyze sentiment, identify common praise and complaints,
    and summarize findings to help customers make informed decisions.""",
    llm=llm,
    verbose=False,
    allow_delegation=True
)

manager_agent = Agent(
    role="Research Coordinator",
    goal="Coordinate product research efforts and ensure comprehensive, high-quality responses",
    backstory="""You are a senior analyst who coordinates research teams.
    You ensure that product recommendations are backed by both expert knowledge
    and real customer feedback. You synthesize information from multiple sources
    to provide the best possible guidance to customers.""",
    llm=llm,
    verbose=False,
    allow_delegation=True
)
