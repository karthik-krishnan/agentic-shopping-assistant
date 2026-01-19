# Agentic Shopping Assistant

A multi-agent product recommendation system built with [CrewAI](https://github.com/crewAIInc/crewAI), demonstrating hierarchical orchestration, guardrails, and fallback handling.

## Architecture

```
                    ┌─────────────────┐
                    │  Manager Agent  │
                    │  (Coordinator)  │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
       ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
       │   Product   │ │  Research   │ │  Synthesis  │
       │   Expert    │ │   Agent     │ │    Task     │
       └─────────────┘ └─────────────┘ └─────────────┘
```

## Agents

| Agent | Role |
|-------|------|
| **Product Expert** | Recommends products based on features and specifications |
| **Research Agent** | Analyzes customer reviews, ratings, and feedback |
| **Manager Agent** | Coordinates tasks and ensures comprehensive responses |

## Features

- **Hierarchical Orchestration** - Manager agent coordinates worker agents
- **Guardrails** - Validates agent responses before proceeding
- **Retry Logic** - Automatic retries with exponential backoff
- **Fallback Responses** - Graceful degradation when agents fail

## Setup

1. **Clone the repository**
   ```bash
   git clone git@github.com:karthik-krishnan/agentic-shopping-assistant.git
   cd agentic-shopping-assistant
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**

   Create a `.env` file with your Azure OpenAI credentials:
   ```
   AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment-name
   AZURE_OPENAI_API_KEY=your-api-key
   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
   AZURE_OPENAI_API_VERSION=2024-02-15-preview
   ```

## Usage

```bash
python main.py
```

## Project Structure

```
├── agent.py        # Agent definitions (Product Expert, Research, Manager)
├── task.py         # Task definitions with guardrails
├── fallback.py     # Retry logic and fallback responses
├── main.py         # Entry point with hierarchical orchestration
├── requirements.txt
└── .env            # Environment variables (not committed)
```

## How It Works

1. **Product Task** - Product Expert identifies products matching criteria
2. **Research Task** - Research Agent gathers customer reviews and feedback
3. **Synthesis Task** - Combines findings into a prioritized recommendation

The Manager Agent coordinates these tasks using CrewAI's hierarchical process, ensuring quality responses through guardrails that validate output before proceeding.

## Requirements

- Python 3.10+
- Azure OpenAI API access
