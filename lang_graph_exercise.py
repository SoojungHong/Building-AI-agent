Here's a basic example of how to create and use an AI agent workflow in LangGraph. This setup illustrates the concepts of state management, conditional branching, and tool integration.

Example: Using LangGraph to Build a Weather Query Agent
Installation: First, you need to install langgraph and any necessary dependencies:

pip install langgraph
Code Example
from langgraph.graph import StateGraph, MessagesState, END, START
from langchain_anthropic import ChatAnthropic  # Example LLM from Anthropic
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool

# Define a sample tool (for illustration)
@tool
def search(query: str) -> str:
    """Mock search function to simulate getting weather information."""
    if "sf" in query.lower() or "san francisco" in query.lower():
        return "It's 60 degrees and foggy in San Francisco."
    return "Sorry, I don't have the weather data for that location."

# Create a list of tools for the agent to use
tools = [search]

# Initialize a model (example: Anthropic Claude)
model = ChatAnthropic(model="claude-3", temperature=0.7).bind_tools(tools)

# Function to determine the next node in the graph
def should_continue(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"  # If the model suggests a tool call, proceed to the tools node
    return END  # Otherwise, end the workflow

# Function to invoke the model
def call_model(state: MessagesState):
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}

# Create the graph
workflow = StateGraph(MessagesState)

# Add nodes to the graph
workflow.add_node("agent", call_model)
tool_node = ToolNode(tools)
workflow.add_node("tools", tool_node)

# Define edges in the graph
workflow.add_edge(START, "agent")  # Start with the agent node
workflow.add_conditional_edges("agent", should_continue)  # Conditional transition
workflow.add_edge("tools", "agent")  # Loop back from tools to the agent

# Compile the graph into a runnable application
app = workflow.compile()

# Test the application
initial_state = {"messages": [HumanMessage(content="What's the weather in SF?")]}
final_state = app.invoke(initial_state)

# Output the final response
print(final_state["messages"][-1].content)
Explanation
Tool Definition: The @tool decorator defines a search function that simulates a tool the agent can use (in this case, fetching weather data).
Model Binding: We use an Anthropic model (ChatAnthropic) and bind the search tool to it.
Graph Workflow:
Nodes: agent calls the model, and tools is where tool usage is handled.
Edges: Control the flow of execution. The workflow starts at agent, checks if a tool should be called, and then loops or ends based on conditions.
State Management: MessagesState is used to track messages and the conversation state, making it easy to handle user queries and model responses.
Features Highlighted
Conditional Logic: The agent decides whether to call a tool or terminate the conversation.
Tool Integration: Allows the model to call predefined functions.
State Management: LangGraph efficiently manages conversation state, even supporting recovery and error handling.
This example illustrates how LangGraph structures a language model application as a graph, which provides flexibility and control over execution flow. LangGraph is particularly useful for applications that need complex workflows, error recovery, or human-in-the-loop interactions.

More Resources
For more advanced examples and documentation, check out the LangGraph GitHub Repository​
GitHub
.
