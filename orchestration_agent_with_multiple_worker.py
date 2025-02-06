"""
Here's how you can implement a multi-agent system using LangChain with an orchestration agent and two worker agents.

🔹 Overview of the Implementation
Orchestration Agent

Receives a user prompt
Defines a list of tasks
Calls the relevant worker agents
Worker Agents

Worker Agent 1: Handles text summarization
Worker Agent 2: Handles sentiment analysis

🔹 Implementation Using LangChain
Here's a Python script demonstrating how to implement multi-agent orchestration:

1️⃣ Install Dependencies
Make sure you have LangChain installed:

bash
Copy
Edit
pip install langchain openai

🔹  How It Works
User provides a prompt
Example: "Can you summarize this paragraph?"
Orchestration Agent determines tasks
If "summarize" is mentioned → Calls Worker Agent 1
If "sentiment" is mentioned → Calls Worker Agent 2
Worker Agents process the request
Worker Agent 1 summarizes the text
Worker Agent 2 performs sentiment analysis
Results are returned to the user

🔹 Example Output
bash
Copy
Edit
> Running summarize_text with input: "LangChain is an AI framework..."
> Running analyze_sentiment with input: "LangChain is an AI framework..."

🔹 Final Results: 
{
  "summary": "LangChain is a framework for developing applications powered by LLMs.",
  "sentiment": "The sentiment is neutral to positive."
}

"""

from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.tools import tool
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
import os

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

### 1️⃣ Worker Agent 1: Text Summarization ###
@tool
def summarize_text(text: str) -> str:
    """Summarizes a given text input."""
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    response = llm([SystemMessage(content="Summarize the following text."), HumanMessage(content=text)])
    return response.content


### 2️⃣ Worker Agent 2: Sentiment Analysis ###
@tool
def analyze_sentiment(text: str) -> str:
    """Analyzes sentiment of a given text."""
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    response = llm([SystemMessage(content="Analyze the sentiment of the following text."), HumanMessage(content=text)])
    return response.content


### 3️⃣ Orchestration Agent ###
def orchestration_agent(user_prompt):
    """
    The Orchestration Agent receives a user prompt,
    determines the necessary tasks, and calls worker agents.
    """
    
    # Determine what tasks are needed
    tasks = []
    if "summarize" in user_prompt.lower():
        tasks.append("summarization")
    if "sentiment" in user_prompt.lower():
        tasks.append("sentiment analysis")
    
    # Initialize the LangChain Agent with memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    agent = initialize_agent(
        tools=[summarize_text, analyze_sentiment],
        llm=ChatOpenAI(model_name="gpt-4", temperature=0),
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        memory=memory
    )
    
    # Execute the required tasks
    results = {}
    for task in tasks:
        if task == "summarization":
            results["summary"] = agent.run(f"Summarize this text: {user_prompt}")
        elif task == "sentiment analysis":
            results["sentiment"] = agent.run(f"Analyze sentiment of this text: {user_prompt}")

    return results


### 4️⃣ Example Usage ###
if __name__ == "__main__":
    user_input = "Can you summarize this paragraph: LangChain is an AI framework for building LLM-powered applications?"
    result = orchestration_agent(user_input)
    print("\n🔹 Final Results:", result)
