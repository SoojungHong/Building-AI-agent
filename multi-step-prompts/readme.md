# Handling Multi-Step Dependent Questions in LangGraph

Complete guide to implementing agents that handle complex queries where subsequent questions depend on previous answers.

## Table of Contents

- [Problem Statement](#problem-statement)
- [Solution Approaches](#solution-approaches)
  - [Approach 1: Sequential Agent with State Management](#approach-1-sequential-agent-with-state-management-recommended)
  - [Approach 2: Agent with Tools (Function Calling)](#approach-2-agent-with-tools-function-calling)
  - [Approach 3: Hierarchical Agent (Sub-Agents)](#approach-3-hierarchical-agent-sub-agents)
  - [Approach 4: Chain-of-Thought with Explicit Planning](#approach-4-chain-of-thought-with-explicit-planning)
- [Comparison of Approaches](#comparison-of-approaches)
- [Recommended: Hybrid Approach](#recommended-hybrid-approach)
- [Best Practices](#best-practices)
- [Testing and Debugging](#testing-and-debugging)

---

## Problem Statement

**Challenge:** Handle prompts with multiple questions where the second question depends on the first answer.

**Example Query:**
> "What is the user's most frequently read news article and based on that which product should be recommended?"

**Requirements:**
1. Answer first question: Find most read article
2. Use the answer from step 1 to answer second question: Recommend product
3. Provide coherent final response combining both answers

---

## Solution Approaches

### Approach 1: Sequential Agent with State Management (Recommended)

Best for: Simple to moderate complexity with clear sequential steps.

**Architecture:**
```
Query → Analyze → Step 1 (Get Article) → Step 2 (Recommend) → Synthesize → Answer
```

**Complete Implementation:**
```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from typing import TypedDict, Annotated, Optional
from operator import add
import json

# Define state to track progress
class AgentState(TypedDict):
    messages: Annotated[list, add]
    user_id: str
    original_query: str
    step: int  # Track which step we're on
    most_read_article: Optional[str]  # Answer to first question
    recommended_product: Optional[str]  # Answer to second question
    intermediate_results: dict  # Store any intermediate data

llm = ChatOpenAI(model="gpt-4", temperature=0)

# Step 1: Analyze the query and extract sub-questions
def query_analyzer(state: AgentState):
    """Break down the complex query into steps."""
    
    system_prompt = """You are a query analyzer. Break down complex queries into sequential steps.
    
    For the query: "What is the user's most frequently read news article and based on that which product should be recommended?"
    
    Identify:
    1. First question: What is the user's most frequently read news article?
    2. Second question (depends on answer 1): Which product should be recommended?
    
    Output JSON:
    {
        "steps": [
            {"step": 1, "question": "...", "depends_on": null},
            {"step": 2, "question": "...", "depends_on": "step_1"}
        ]
    }
    """
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=state["original_query"])
    ]
    
    response = llm.invoke(messages)
    
    try:
        analysis = json.loads(response.content)
        return {
            "intermediate_results": {"analysis": analysis},
            "step": 1,
            "messages": [HumanMessage(content=f"Query analyzed: {len(analysis['steps'])} steps identified")]
        }
    except:
        return {
            "step": 1,
            "messages": [HumanMessage(content="Query analysis complete")]
        }

# Step 2: Answer the first question
def answer_first_question(state: AgentState):
    """Find the user's most frequently read news article."""
    
    user_id = state["user_id"]
    
    # Simulate database query
    # In production, replace with actual database call
    user_reading_data = get_user_reading_history(user_id)
    
    most_read = user_reading_data["most_frequent_article"]
    
    return {
        "most_read_article": most_read,
        "step": 2,
        "intermediate_results": {
            **state.get("intermediate_results", {}),
            "reading_data": user_reading_data
        },
        "messages": [HumanMessage(content=f"Most read article: {most_read}")]
    }

# Step 3: Answer the second question using first answer
def recommend_product(state: AgentState):
    """Recommend product based on most read article."""
    
    most_read_article = state["most_read_article"]
    user_id = state["user_id"]
    
    system_prompt = f"""You are a product recommendation specialist.
    
    User's most frequently read article: {most_read_article}
    
    Based on the article topic and content, recommend a relevant product.
    Consider the user's interests demonstrated by this article.
    
    Provide:
    1. Product name
    2. Why it's relevant
    3. Confidence score (0-1)
    """
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Recommend a product for user {user_id}")
    ]
    
    response = llm.invoke(messages)
    
    return {
        "recommended_product": response.content,
        "step": 3,
        "messages": [HumanMessage(content=f"Recommendation: {response.content}")]
    }

# Step 4: Synthesize final answer
def synthesize_answer(state: AgentState):
    """Combine both answers into a coherent response."""
    
    system_prompt = f"""Create a comprehensive answer combining these findings:
    
    1. Most read article: {state['most_read_article']}
    2. Product recommendation: {state['recommended_product']}
    
    Provide a natural, conversational response that answers the original query:
    "{state['original_query']}"
    """
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content="Synthesize the final answer")
    ]
    
    response = llm.invoke(messages)
    
    return {
        "messages": [HumanMessage(content=response.content)]
    }

# Router: Decide which step to execute next
def route_step(state: AgentState):
    """Route to the appropriate node based on current step."""
    step = state.get("step", 0)
    
    if step == 0:
        return "analyze"
    elif step == 1:
        return "first_question"
    elif step == 2:
        return "recommend"
    elif step == 3:
        return "synthesize"
    else:
        return END

# Build the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("analyze", query_analyzer)
workflow.add_node("first_question", answer_first_question)
workflow.add_node("recommend", recommend_product)
workflow.add_node("synthesize", synthesize_answer)

# Set entry point
workflow.set_entry_point("analyze")

# Add conditional edges
workflow.add_conditional_edges(
    "analyze",
    route_step,
    {
        "first_question": "first_question",
        END: END
    }
)

workflow.add_conditional_edges(
    "first_question",
    route_step,
    {
        "recommend": "recommend",
        END: END
    }
)

workflow.add_conditional_edges(
    "recommend",
    route_step,
    {
        "synthesize": "synthesize",
        END: END
    }
)

workflow.add_edge("synthesize", END)

# Compile with memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Helper function to simulate database
def get_user_reading_history(user_id: str):
    """Simulate fetching user reading history from database."""
    # Replace with actual database query
    return {
        "most_frequent_article": "Understanding Cryptocurrency Markets",
        "read_count": 15,
        "category": "Finance/Technology",
        "last_read": "2024-12-08"
    }

# Usage
def process_complex_query(query: str, user_id: str, thread_id: str = "default"):
    """Process a complex multi-step query."""
    
    config = {"configurable": {"thread_id": thread_id}}
    
    initial_state = {
        "messages": [],
        "user_id": user_id,
        "original_query": query,
        "step": 0,
        "most_read_article": None,
        "recommended_product": None,
        "intermediate_results": {}
    }
    
    result = app.invoke(initial_state, config=config)
    
    # Return the final answer
    final_message = result["messages"][-1].content
    return {
        "answer": final_message,
        "most_read_article": result["most_read_article"],
        "recommended_product": result["recommended_product"],
        "intermediate_results": result["intermediate_results"]
    }

# Test
result = process_complex_query(
    query="What is the user's most frequently read news article and based on that which product should be recommended?",
    user_id="user-123",
    thread_id="query-001"
)

print(result["answer"])
```

**Pros:**
- ✅ Clear, explicit flow
- ✅ Easy to debug
- ✅ Good for fixed sequential steps
- ✅ Full control over state

**Cons:**
- ❌ Less flexible for dynamic queries
- ❌ Requires predefined steps

---

### Approach 2: Agent with Tools (Function Calling)

Best for: Dynamic question routing where the LLM decides the execution flow.

**Architecture:**
```
Query → LLM Agent → Tool 1 (Get Article) → Tool 2 (Recommend) → Final Answer
                ↓
            [Decides which tools to use and when]
```

**Complete Implementation:**
```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated

# Define tools for each sub-question
@tool
def get_most_read_article(user_id: Annotated[str, "The user ID to query"]) -> str:
    """Get the user's most frequently read news article.
    
    Returns the title and details of the most read article.
    """
    # Simulate database query
    user_data = {
        "user-123": {
            "most_read": "Understanding Cryptocurrency Markets",
            "category": "Finance/Technology",
            "read_count": 15
        }
    }
    
    data = user_data.get(user_id, {})
    article = data.get("most_read", "No data available")
    category = data.get("category", "Unknown")
    count = data.get("read_count", 0)
    
    return f"Most read article: '{article}' (Category: {category}, Read {count} times)"

@tool
def recommend_product_based_on_article(
    article_title: Annotated[str, "The article title to base recommendation on"],
    user_id: Annotated[str, "The user ID"]
) -> str:
    """Recommend a product based on the article the user reads most.
    
    Analyzes the article topic and recommends a relevant product.
    """
    # Simulate product recommendation logic
    recommendations = {
        "Understanding Cryptocurrency Markets": {
            "product": "Crypto Portfolio Tracker Pro",
            "reason": "Based on your interest in cryptocurrency, this tool helps track and manage crypto investments",
            "confidence": 0.92
        },
        "Climate Change Solutions": {
            "product": "Green Energy Monitor",
            "reason": "Aligns with your interest in environmental topics",
            "confidence": 0.88
        }
    }
    
    rec = recommendations.get(
        article_title,
        {
            "product": "News Digest Premium",
            "reason": "General news reader upgrade",
            "confidence": 0.65
        }
    )
    
    return f"Recommended Product: {rec['product']}\nReason: {rec['reason']}\nConfidence: {rec['confidence']}"

# Create agent with tools
llm = ChatOpenAI(model="gpt-4", temperature=0)
tools = [get_most_read_article, recommend_product_based_on_article]

memory = MemorySaver()
agent = create_react_agent(
    llm,
    tools,
    checkpointer=memory,
    state_modifier="""You are a helpful assistant that answers complex questions step by step.

When faced with a multi-part question:
1. Break it down into steps
2. Execute each step in order
3. Use the result from the first step to inform the second step
4. Provide a comprehensive final answer

Always use the available tools to get accurate data."""
)

# Usage
def chat_with_agent(query: str, user_id: str, thread_id: str = "default"):
    """Chat with the agent using tools."""
    
    config = {"configurable": {"thread_id": thread_id}}
    
    # Add user_id context to the query
    enhanced_query = f"User ID: {user_id}\n\nQuery: {query}"
    
    result = agent.invoke(
        {"messages": [("user", enhanced_query)]},
        config=config
    )
    
    return result["messages"][-1].content

# Test
response = chat_with_agent(
    query="What is my most frequently read news article and based on that which product should you recommend?",
    user_id="user-123",
    thread_id="session-001"
)

print(response)
```

**Pros:**
- ✅ Flexible - LLM decides execution flow
- ✅ Handles dynamic queries well
- ✅ Less code to maintain
- ✅ Easy to add new tools

**Cons:**
- ❌ Less predictable
- ❌ Higher LLM costs (more calls)
- ❌ Harder to debug

---

### Approach 3: Hierarchical Agent (Sub-Agents)

Best for: Complex domain logic with specialized expertise needed for each step.

**Architecture:**
```
Coordinator Agent
    ├─→ Reading Analysis Sub-Agent (Step 1)
    └─→ Product Recommendation Sub-Agent (Step 2, uses Step 1 result)
```

**Complete Implementation:**
```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from typing import TypedDict, Annotated, List
from operator import add

class CoordinatorState(TypedDict):
    messages: Annotated[list, add]
    user_id: str
    original_query: str
    sub_results: dict
    final_answer: str

# Sub-Agent 1: Reading Analysis Agent
class ReadingAnalysisAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    def analyze(self, user_id: str) -> dict:
        """Analyze user's reading patterns."""
        
        # Fetch data (simulate)
        reading_data = self._fetch_reading_data(user_id)
        
        # Analyze with LLM
        system_prompt = """You are a reading pattern analyst.
        Analyze the user's reading history and identify:
        1. Most frequently read article
        2. Reading patterns
        3. Interest categories
        
        Return structured data."""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Reading data: {reading_data}")
        ]
        
        response = self.llm.invoke(messages)
        
        return {
            "most_read_article": reading_data["most_frequent"],
            "category": reading_data["category"],
            "analysis": response.content
        }
    
    def _fetch_reading_data(self, user_id: str):
        """Simulate database fetch."""
        return {
            "most_frequent": "Understanding Cryptocurrency Markets",
            "category": "Finance/Technology",
            "total_articles": 150,
            "read_count_top": 15
        }

# Sub-Agent 2: Product Recommendation Agent
class ProductRecommendationAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    def recommend(self, article_info: dict, user_id: str) -> dict:
        """Recommend product based on article."""
        
        article_title = article_info["most_read_article"]
        category = article_info["category"]
        
        system_prompt = f"""You are a product recommendation specialist.
        
        User's reading profile:
        - Most read article: {article_title}
        - Category: {category}
        
        Recommend the most relevant product with:
        1. Product name
        2. Detailed reason
        3. Expected benefits
        4. Confidence score
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Recommend product for user {user_id}")
        ]
        
        response = self.llm.invoke(messages)
        
        return {
            "recommendation": response.content,
            "based_on_article": article_title
        }

# Coordinator Agent
def coordinator_node(state: CoordinatorState):
    """Coordinate between sub-agents."""
    
    user_id = state["user_id"]
    
    # Step 1: Invoke Reading Analysis Agent
    reading_agent = ReadingAnalysisAgent()
    reading_result = reading_agent.analyze(user_id)
    
    # Step 2: Invoke Product Recommendation Agent (using result from Step 1)
    recommendation_agent = ProductRecommendationAgent()
    recommendation_result = recommendation_agent.recommend(reading_result, user_id)
    
    # Step 3: Synthesize final answer
    final_answer = f"""Based on your reading history:

**Most Read Article:** {reading_result['most_read_article']}
**Category:** {reading_result['category']}

**Reading Analysis:**
{reading_result['analysis']}

**Product Recommendation:**
{recommendation_result['recommendation']}

This recommendation is specifically tailored to your demonstrated interest in {reading_result['category'].lower()} topics."""
    
    return {
        "sub_results": {
            "reading_analysis": reading_result,
            "product_recommendation": recommendation_result
        },
        "final_answer": final_answer,
        "messages": [HumanMessage(content=final_answer)]
    }

# Build coordinator graph
workflow = StateGraph(CoordinatorState)
workflow.add_node("coordinator", coordinator_node)
workflow.set_entry_point("coordinator")
workflow.add_edge("coordinator", END)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Usage
def process_with_hierarchical_agents(query: str, user_id: str, thread_id: str = "default"):
    """Process query using hierarchical agent architecture."""
    
    config = {"configurable": {"thread_id": thread_id}}
    
    result = app.invoke(
        {
            "messages": [],
            "user_id": user_id,
            "original_query": query,
            "sub_results": {},
            "final_answer": ""
        },
        config=config
    )
    
    return result["final_answer"]

# Test
answer = process_with_hierarchical_agents(
    query="What is my most frequently read news article and based on that which product should you recommend?",
    user_id="user-123",
    thread_id="hierarchical-001"
)

print(answer)
```

**Pros:**
- ✅ Modular and maintainable
- ✅ Each sub-agent can be optimized independently
- ✅ Good for complex domain logic
- ✅ Easy to add new sub-agents

**Cons:**
- ❌ More complex architecture
- ❌ Higher overhead
- ❌ Overkill for simple queries

---

### Approach 4: Chain-of-Thought with Explicit Planning

Best for: Queries with unknown complexity that need dynamic planning.

**Architecture:**
```
Query → Create Plan → Execute Step 1 → Execute Step 2 → ... → Synthesize
```

**Complete Implementation:**
```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from typing import TypedDict, Annotated, List
from operator import add
import json

class PlanExecuteState(TypedDict):
    messages: Annotated[list, add]
    user_id: str
    original_query: str
    plan: List[dict]
    current_step: int
    step_results: dict
    final_answer: str

llm = ChatOpenAI(model="gpt-4", temperature=0)

# Node 1: Create Plan
def create_plan(state: PlanExecuteState):
    """Create an execution plan for the complex query."""
    
    system_prompt = """You are a query planner. Break down complex queries into executable steps.
    
    For each step, specify:
    1. step_id: unique identifier
    2. description: what to do
    3. depends_on: which previous step(s) this depends on (null if none)
    4. tool_to_use: which function/tool to call
    
    Output ONLY valid JSON array."""
    
    user_prompt = f"""Query: {state['original_query']}
    User ID: {state['user_id']}
    
    Create a step-by-step plan."""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    
    try:
        # Clean response and parse JSON
        content = response.content.strip()
        if content.startswith("```json"):
            content = content[7:-3]
        elif content.startswith("```"):
            content = content[3:-3]
        
        plan = json.loads(content)
        
        return {
            "plan": plan,
            "current_step": 0,
            "messages": [HumanMessage(content=f"Created plan with {len(plan)} steps")]
        }
    except Exception as e:
        # Fallback plan
        default_plan = [
            {
                "step_id": 1,
                "description": "Get user's most read article",
                "depends_on": None,
                "tool_to_use": "get_reading_history"
            },
            {
                "step_id": 2,
                "description": "Recommend product based on article",
                "depends_on": [1],
                "tool_to_use": "recommend_product"
            }
        ]
        
        return {
            "plan": default_plan,
            "current_step": 0,
            "messages": [HumanMessage(content=f"Created default plan: {e}")]
        }

# Node 2: Execute Step
def execute_step(state: PlanExecuteState):
    """Execute the current step in the plan."""
    
    plan = state["plan"]
    current_step = state["current_step"]
    
    if current_step >= len(plan):
        return {"current_step": current_step}  # Done
    
    step = plan[current_step]
    step_id = step["step_id"]
    description = step["description"]
    depends_on = step.get("depends_on")
    
    # Get dependencies
    dependency_context = ""
    if depends_on:
        for dep_id in depends_on:
            dep_result = state["step_results"].get(f"step_{dep_id}")
            if dep_result:
                dependency_context += f"\nResult from step {dep_id}: {dep_result}\n"
    
    # Execute step with context
    system_prompt = f"""You are executing step {step_id} of a plan.
    
    Step description: {description}
    
    Previous results:
    {dependency_context if dependency_context else "No dependencies"}
    
    User ID: {state['user_id']}
    
    Execute this step and provide the result."""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Execute: {description}")
    ]
    
    response = llm.invoke(messages)
    
    # Store result
    new_step_results = state.get("step_results", {})
    new_step_results[f"step_{step_id}"] = response.content
    
    return {
        "step_results": new_step_results,
        "current_step": current_step + 1,
        "messages": [HumanMessage(content=f"Step {step_id} completed: {response.content[:100]}...")]
    }

# Node 3: Check if Done
def check_completion(state: PlanExecuteState):
    """Check if all steps are completed."""
    current_step = state["current_step"]
    total_steps = len(state["plan"])
    
    if current_step >= total_steps:
        return "synthesize"
    else:
        return "execute"

# Node 4: Synthesize Answer
def synthesize_final_answer(state: PlanExecuteState):
    """Combine all step results into final answer."""
    
    system_prompt = f"""Synthesize a comprehensive answer from these step results:
    
    Original Query: {state['original_query']}
    
    Step Results:
    {json.dumps(state['step_results'], indent=2)}
    
    Provide a natural, comprehensive answer."""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content="Synthesize final answer")
    ]
    
    response = llm.invoke(messages)
    
    return {
        "final_answer": response.content,
        "messages": [HumanMessage(content=response.content)]
    }

# Build graph
workflow = StateGraph(PlanExecuteState)

workflow.add_node("plan", create_plan)
workflow.add_node("execute", execute_step)
workflow.add_node("synthesize", synthesize_final_answer)

workflow.set_entry_point("plan")
workflow.add_edge("plan", "execute")

workflow.add_conditional_edges(
    "execute",
    check_completion,
    {
        "execute": "execute",
        "synthesize": "synthesize"
    }
)

workflow.add_edge("synthesize", END)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Usage
def process_with_planning(query: str, user_id: str, thread_id: str = "default"):
    """Process query using plan-execute approach."""
    
    config = {"configurable": {"thread_id": thread_id}}
    
    result = app.invoke(
        {
            "messages": [],
            "user_id": user_id,
            "original_query": query,
            "plan": [],
            "current_step": 0,
            "step_results": {},
            "final_answer": ""
        },
        config=config
    )
    
    return {
        "answer": result["final_answer"],
        "plan": result["plan"],
        "step_results": result["step_results"]
    }

# Test
result = process_with_planning(
    query="What is my most frequently read news article and based on that which product should you recommend?",
    user_id="user-123",
    thread_id="plan-001"
)

print("=" * 50)
print("PLAN:")
print(json.dumps(result["plan"], indent=2))
print("\n" + "=" * 50)
print("FINAL ANSWER:")
print(result["answer"])
```

**Pros:**
- ✅ Handles unknown complexity
- ✅ Most flexible approach
- ✅ Can adapt to any query structure
- ✅ Shows reasoning transparently

**Cons:**
- ❌ Slowest approach
- ❌ Highest LLM costs
- ❌ More complex to debug
- ❌ Planning can fail

---

## Comparison of Approaches

| Approach | Best For | Complexity | Flexibility | Performance | LLM Costs | Debuggability |
|----------|----------|------------|-------------|-------------|-----------|---------------|
| **Sequential State** | Fixed 2-3 step queries | Low | Medium | ⚡⚡⚡ Fast | 💰 Low | ⭐⭐⭐ Easy |
| **Agent with Tools** | Dynamic routing | Medium | High | ⚡⚡ Medium | 💰💰 Medium | ⭐⭐ Medium |
| **Hierarchical Agents** | Domain-specific logic | High | High | ⚡⚡ Medium | 💰💰 Medium | ⭐⭐ Medium |
| **Plan-Execute** | Unknown complexity | High | Highest | ⚡ Slow | 💰💰💰 High | ⭐ Hard |

### When to Use Each:

**Use Sequential State when:**
- You have 2-3 clear sequential steps
- Flow is predictable
- Performance is critical
- Budget is limited

**Use Agent with Tools when:**
- Questions are dynamic
- You want LLM to decide flow
- You have good tools/functions
- You need flexibility

**Use Hierarchical Agents when:**
- You have complex domain logic
- Each step needs specialized expertise
- Modularity is important
- You're building a large system

**Use Plan-Execute when:**
- Query complexity is unknown
- You need maximum flexibility
- Transparency of reasoning is important
- Performance is not critical

---

## Recommended: Hybrid Approach

For production systems, combine approaches for best results.
```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from typing import TypedDict, Annotated, Optional, List
from operator import add
import json

class HybridAgentState(TypedDict):
    messages: Annotated[list, add]
    user_id: str
    original_query: str
    query_type: str  # "simple", "sequential", "complex"
    steps: List[dict]
    current_step: int
    results: dict
    final_answer: str

llm = ChatOpenAI(model="gpt-4", temperature=0)

# Classify query complexity
def classify_query(state: HybridAgentState):
    """Determine query type and complexity."""
    
    system_prompt = """Classify this query:
    
    Types:
    - "simple": Single question, no dependencies
    - "sequential": Multiple questions where later ones depend on earlier answers
    - "complex": Requires planning and multiple data sources
    
    Output only the type."""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=state["original_query"])
    ]
    
    response = llm.invoke(messages)
    query_type = response.content.strip().lower()
    
    return {
        "query_type": query_type,
        "messages": [HumanMessage(content=f"Query classified as: {query_type}")]
    }

# Route based on complexity
def route_by_complexity(state: HybridAgentState):
    """Route to appropriate handler."""
    query_type = state.get("query_type", "simple")
    
    if query_type == "simple":
        return "simple_handler"
    elif query_type == "sequential":
        return "sequential_handler"
    else:
        return "complex_handler"

# Simple handler
def simple_handler(state: HybridAgentState):
    """Handle simple queries directly."""
    
    messages = [
        SystemMessage(content="Answer this query directly and concisely."),
        HumanMessage(content=f"User {state['user_id']}: {state['original_query']}")
    ]
    
    response = llm.invoke(messages)
    
    return {
        "final_answer": response.content,
        "messages": [HumanMessage(content=response.content)]
    }

# Sequential handler (your use case)
def sequential_handler(state: HybridAgentState):
    """Handle sequential dependent queries."""
    
    user_id = state["user_id"]
    
    # Step 1: Get most read article
    reading_data = get_user_reading_history(user_id)
    most_read = reading_data["most_frequent_article"]
    
    step1_result = f"Most frequently read article: {most_read}"
    
    # Step 2: Recommend product based on article
    system_prompt = f"""Based on the user's most read article: "{most_read}"
    
    Recommend a relevant product with reasoning."""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Recommend product for user {user_id}")
    ]
    
    response = llm.invoke(messages)
    
    final_answer = f"""{step1_result}

Product Recommendation:
{response.content}"""
    
    return {
        "results": {
            "step1": step1_result,
            "step2": response.content
        },
        "final_answer": final_answer,
        "messages": [HumanMessage(content=final_answer)]
    }

# Complex handler
def complex_handler(state: HybridAgentState):
    """Handle complex queries with planning."""
    
    # Create and execute plan (simplified)
    plan_prompt = f"""Create a 3-step plan for: {state['original_query']}
    
    User ID: {state['user_id']}"""
    
    messages = [
        SystemMessage(content="You are a query planner."),
        HumanMessage(content=plan_prompt)
    ]
    
    response = llm.invoke(messages)
    
    return {
        "final_answer": response.content,
        "messages": [HumanMessage(content=response.content)]
    }

# Build graph
workflow = StateGraph(HybridAgentState)

workflow.add_node("classify", classify_query)
workflow.add_node("simple_handler", simple_handler)
workflow.add_node("sequential_handler", sequential_handler)
workflow.add_node("complex_handler", complex_handler)

workflow.set_entry_point("classify")

workflow.add_conditional_edges(
    "classify",
    route_by_complexity,
    {
        "simple_handler": "simple_handler",
        "sequential_handler": "sequential_handler",
        "complex_handler": "complex_handler"
    }
)

workflow.add_edge("simple_handler", END)
workflow.add_edge("sequential_handler", END)
workflow.add_edge("complex_handler", END)

# Compile with persistent memory
memory = SqliteSaver.from_conn_string("agent_memory.db")
app = workflow.compile(checkpointer=memory)

def get_user_reading_history(user_id: str):
    """Simulate database query."""
    return {
        "most_frequent_article": "Understanding Cryptocurrency Markets",
        "read_count": 15,
        "category": "Finance/Technology"
    }

# Main interface
def ask_agent(query: str, user_id: str, thread_id: str = None):
    """Ask the hybrid agent a question."""
    
    if thread_id is None:
        import uuid
        thread_id = str(uuid.uuid4())
    
    config = {"configurable": {"thread_id": thread_id}}
    
    result = app.invoke(
        {
            "messages": [],
            "user_id": user_id,
            "original_query": query,
            "query_type": "",
            "steps": [],
            "current_step": 0,
            "results": {},
            "final_answer": ""
        },
        config=config
    )
    
    return result["final_answer"]

# Test with your specific query
answer = ask_agent(
    query="What is my most frequently read news article and based on that which product should you recommend?",
    user_id="user-123"
)

print(answer)
```

**Why Hybrid?**
- ✅ Automatically routes to best approach
- ✅ Cost-effective (simple queries don't use complex logic)
- ✅ Flexible and maintainable
- ✅ Production-ready

---

## Best Practices

### 1. State Management
```python
# ✅ Good: Clear state structure
class AgentState(TypedDict):
    # Always include
    messages: Annotated[list, add]
    user_id: str
    original_query: str
    
    # Step tracking
    current_step: int
    
    # Results from each step
    step_results: dict
    
    # Final answer
    final_answer: str
    
    # Optional metadata
    metadata: dict
```

### 2. Error Handling
```python
def safe_step_execution(state: AgentState):
    """Execute step with error handling."""
    try:
        # Your logic here
        result = execute_logic(state)
        return result
    except Exception as e:
        # Log error
        print(f"Error in step: {e}")
        
        # Return error state
        return {
            "error": str(e),
            "messages": [HumanMessage(content=f"Error: {e}")]
        }
```

### 3. Logging and Debugging
```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def logged_step(state: AgentState):
    """Step with logging."""
    logger.info(f"Executing step {state['current_step']}")
    logger.debug(f"State: {state}")
    
    result = execute_step(state)
    
    logger.info(f"Step completed. Result: {result}")
    return result
```

### 4. Testing
```python
import pytest

def test_sequential_agent():
    """Test sequential query handling."""
    result = process_complex_query(
        query="What is my most read article and recommend product?",
        user_id="test-user-123",
        thread_id="test-001"
    )
    
    assert result["most_read_article"] is not None
    assert result["recommended_product"] is not None
    assert len(result["answer"]) > 0

def test_dependency_passing():
    """Test that second step uses first step result."""
    # Mock the state
    state = {
        "user_id": "test-123",
        "most_read_article": "Test Article",
        "step": 2
    }
    
    result = recommend_product(state)
    
    # Verify the recommendation references the article
    assert "Test Article" in str(result)
```

### 5. Performance Optimization
```python
# Cache expensive operations
from functools import lru_cache

@lru_cache(maxsize=100)
def get_user_reading_history(user_id: str):
    """Cached reading history lookup."""
    # Expensive database operation
    return fetch_from_db(user_id)

# Use async for I/O operations
async def async_database_call(user_id: str):
    """Async database call."""
    async with database.connection() as conn:
        result = await conn.fetch(user_id)
        return result
```

### 6. Monitoring
```python
import time
from datetime import datetime

def monitored_step(state: AgentState):
    """Step with monitoring."""
    start_time = time.time()
    
    try:
        result = execute_step(state)
        
        # Log metrics
        duration = time.time() - start_time
        log_metric("step_duration", duration)
        log_metric("step_success", 1)
        
        return result
    except Exception as e:
        log_metric("step_failure", 1)
        raise
```

---

## Testing and Debugging

### Visualize Graph
```python
from IPython.display import Image

# Generate graph visualization
Image(app.get_graph().draw_mermaid_png())
```

### Debug Step by Step
```python
def debug_agent(query: str, user_id: str):
    """Run agent with detailed logging."""
    
    config = {"configurable": {"thread_id": "debug-001"}}
    
    # Enable verbose mode
    state = {
        "messages": [],
        "user_id": user_id,
        "original_query": query,
        "step": 0,
        "most_read_article": None,
        "recommended_product": None,
        "intermediate_results": {}
    }
    
    # Stream results
    for chunk in app.stream(state, config=config):
        print("=" * 50)
        print(f"Step: {chunk}")
        print("=" * 50)
```

### Test Individual Nodes
```python
def test_first_question_node():
    """Test individual node in isolation."""
    
    mock_state = {
        "user_id": "test-123",
        "step": 1,
        "messages": [],
        "intermediate_results": {}
    }
    
    result = answer_first_question(mock_state)
    
    assert "most_read_article" in result
    assert result["step"] == 2
```

---

## Common Issues and Solutions

### Issue 1: State Not Persisting Between Steps

**Problem:**
```python
# State gets lost between nodes
```

**Solution:**
```python
# Always return all state fields
def my_node(state: AgentState):
    return {
        "messages": [new_message],
        "step": state["step"] + 1,
        # Include other unchanged fields if needed
        "user_id": state["user_id"]
    }
```

### Issue 2: Dependencies Not Available

**Problem:**
```python
# Step 2 can't access Step 1 result
```

**Solution:**
```python
# Store results in state
def step1(state):
    result = do_work()
    return {
        "step_results": {
            "step1": result
        },
        "current_step": 2
    }

def step2(state):
    # Access step1 result
    previous_result = state["step_results"]["step1"]
    # Use it...
```

### Issue 3: Infinite Loops

**Problem:**
```python
# Agent keeps routing back to same node
```

**Solution:**
```python
# Add max steps or clear exit conditions
def route_with_safety(state):
    if state["current_step"] > MAX_STEPS:
        return END
    
    # Normal routing logic
    if condition:
        return "next_node"
```

---

## Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Tools](https://python.langchain.com/docs/modules/agents/tools/)
- [State Management Guide](https://langchain-ai.github.io/langgraph/concepts/low_level/)
- [LangGraph Examples](https://github.com/langchain-ai/langgraph/tree/main/examples)

---

## Quick Reference

### Minimal Sequential Agent Template
```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated
from operator import add

class State(TypedDict):
    messages: Annotated[list, add]
    step1_result: str
    step2_result: str

llm = ChatOpenAI(model="gpt-4")

def step1(state):
    result = "Step 1 answer"
    return {"step1_result": result}

def step2(state):
    # Use step1 result
    prev = state["step1_result"]
    result = f"Step 2 using: {prev}"
    return {"step2_result": result}

workflow = StateGraph(State)
workflow.add_node("step1", step1)
workflow.add_node("step2", step2)
workflow.set_entry_point("step1")
workflow.add_edge("step1", "step2")
workflow.add_edge("step2", END)

app = workflow.compile(checkpointer=MemorySaver())
```

---

## Contributing

Found an issue or have a suggestion? Please open an issue or submit a pull request.

---

## License

This guide is provided as-is for educational purposes.

---

**Last Updated:** December 2024

**Version:** 1.0
