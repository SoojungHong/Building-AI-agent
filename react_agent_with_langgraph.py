"""
a complete LangGraph React Agent example using LangChain's tooling. This will show how to implement a ReAct-style agent as a graph, using:

Reasoning steps (LLM)

Tools (e.g., calculator)

A loop that lets the agent act, observe, and repeat until it reaches a final answer

✅ What This Example Will Do
Use OpenAI LLM for reasoning.

Add a calculator tool.

Use LangGraph to build a graph where:

reason node figures out what to do next.

action node runs a tool.

decide node determines if we're done.

Graph loops until we stop.

🧠 How It Works
call_agent: Asks the LLM what to do (e.g., "Use calculator").

run_tool: Executes that tool.

should_continue: If tool name is "Final Answer", we stop. Otherwise, loop again.

This is how LangGraph brings control flow (like loops and conditionals) into LLM workflows — useful for structured reasoning and React agents.

"""

# install 
pip install langchain langgraph openai

#  ReAct Agent with LangGraph (Full Code)

from langchain.chat_models import ChatOpenAI
from langchain.agents import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor, AgentExecutorBuilder

# 1. Define a simple calculator tool
@tool
def calculator(expression: str) -> str:
    """Evaluates a math expression"""
    try:
        return str(eval(expression))
    except:
        return "Error evaluating expression"

tools = [calculator]

# 2. LLM (Chat model)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# 3. Tool executor handles actual tool calls
tool_executor = ToolExecutor(tools)

# 4. Create the agent executor (ReAct-style)
agent_builder = AgentExecutorBuilder.from_llm_and_tools(llm=llm, tools=tools)
agent_executor = agent_builder.build()

# 5. LangGraph state schema
class AgentState(dict):
    pass

# 6. Define graph nodes
def call_agent(state: AgentState) -> AgentState:
    result = agent_executor.invoke({"input": state["input"], "intermediate_steps": state.get("intermediate_steps", [])})
    state["intermediate_steps"] = result.get("intermediate_steps", [])
    state["agent_action"] = result["agent_action"]
    return state

def run_tool(state: AgentState) -> AgentState:
    action = state["agent_action"]
    observation = tool_executor.invoke(action)
    state["intermediate_steps"].append((action, observation))
    return state

def should_continue(state: AgentState) -> str:
    if state["agent_action"].tool == "Final Answer":
        return END
    return "run_tool"

# 7. Build the graph
workflow = StateGraph(AgentState)

workflow.add_node("call_agent", call_agent)
workflow.add_node("run_tool", run_tool)
workflow.set_entry_point("call_agent")
workflow.add_edge("call_agent", should_continue)
workflow.add_edge("run_tool", "call_agent")

# 8. Compile and run
graph = workflow.compile()

response = graph.invoke({"input": "What is 2 + 2 * 3?"})
print(response)
