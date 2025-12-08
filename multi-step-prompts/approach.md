# Multi-Step Dependent Questions in LangGraph - Complete Guide

Handle queries where Question 2 depends on Answer 1 using LangGraph agents.

## 📋 Table of Contents

- [Problem Overview](#problem-overview)
- [Solution Approaches](#solution-approaches)
  - [Approach 1: Sequential State Management (RECOMMENDED)](#approach-1-sequential-state-management-recommended)
  - [Approach 2: Using Tools (More Flexible)](#approach-2-using-tools-more-flexible)
  - [Approach 3: Async for Better Performance](#approach-3-async-for-better-performance)
- [Complete Production Example](#complete-production-example)
- [Testing Your Implementation](#testing-your-implementation)
- [Quick Comparison](#quick-comparison)
- [Key Takeaways](#key-takeaways)
- [FAQ](#faq)

---

## Problem Overview

**Challenge:** Handle queries where Question 2 depends on Answer 1

**Example Query:**
```
"What is the user's most frequently read news article 
and based on that which product should be recommended?"
```

**Required Flow:**
```
Question 1 → Answer 1 → Question 2 (uses Answer 1) → Final Answer
```

**Key Requirement:** Step 2 must have access to Step 1's result.

---

## Solution Approaches

### ⭐ Approach 1: Sequential State Management (RECOMMENDED)

**Best for:** Your exact use case - 2 dependent questions with clear sequential flow.

**Architecture Diagram:**
```
┌─────────────┐
│   Query     │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│  Node 1: Get        │
│  Most Read Article  │
└──────┬──────────────┘
       │ (stores result in state)
       ▼
┌─────────────────────┐
│  Node 2: Recommend  │
│  Product            │ ← Uses result from Node 1
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Node 3: Synthesize │
│  Final Answer       │
└──────┬──────────────┘
       │
       ▼
    Answer
```

**Complete Implementation:**
```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from typing import TypedDict, Optional
import json

# ============================================================================
# STEP 1: Define State
# ============================================================================
class AgentState(TypedDict):
    user_id: str
    original_query: str
    
    # Results from each step
    most_read_article: Optional[str]  # Answer to Q1
    recommended_product: Optional[str]  # Answer to Q2
    
    # Final combined answer
    final_answer: str

# ============================================================================
# STEP 2: Initialize LLM
# ============================================================================
llm = ChatOpenAI(model="gpt-4", temperature=0)

# ============================================================================
# STEP 3: Node 1 - Answer First Question
# ============================================================================
def get_most_read_article(state: AgentState):
    """
    Question 1: What is the user's most frequently read article?
    """
    user_id = state["user_id"]
    
    # Fetch from database (replace with your actual DB call)
    reading_data = fetch_user_reading_history(user_id)
    
    most_read = reading_data.get("most_frequent_article", "Unknown")
    category = reading_data.get("category", "General")
    read_count = reading_data.get("read_count", 0)
    
    # Format answer
    answer = f"'{most_read}' (Category: {category}, Read {read_count} times)"
    
    print(f"✓ Question 1 answered: {answer}")
    
    return {
        "most_read_article": most_read
    }

# ============================================================================
# STEP 4: Node 2 - Answer Second Question (Uses Answer from Node 1)
# ============================================================================
def recommend_product_based_on_article(state: AgentState):
    """
    Question 2: Based on the most read article, what product to recommend?
    This DEPENDS on the answer from Question 1.
    """
    most_read_article = state["most_read_article"]  # ← Use result from Q1
    user_id = state["user_id"]
    
    # Use LLM to generate recommendation
    system_prompt = f"""You are a product recommendation expert.
    
The user's most frequently read article is: "{most_read_article}"

Based on this article topic and the user's demonstrated interest, recommend ONE relevant product.

Provide:
1. Product name
2. Why it's relevant (2-3 sentences)
3. Expected benefit for the user
4. Confidence score (0.0 to 1.0)

Format as JSON."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Recommend a product for user {user_id}")
    ]
    
    response = llm.invoke(messages)
    
    print(f"✓ Question 2 answered based on: {most_read_article}")
    
    return {
        "recommended_product": response.content
    }

# ============================================================================
# STEP 5: Node 3 - Synthesize Final Answer
# ============================================================================
def create_final_answer(state: AgentState):
    """
    Combine both answers into a natural, comprehensive response.
    """
    most_read = state["most_read_article"]
    recommendation = state["recommended_product"]
    original_query = state["original_query"]
    
    system_prompt = f"""Create a natural, conversational answer to this query:
"{original_query}"

Information available:
1. Most read article: {most_read}
2. Product recommendation: {recommendation}

Combine these into a friendly, helpful response that directly answers the original question."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content="Create the final answer")
    ]
    
    response = llm.invoke(messages)
    final_text = response.content
    
    print(f"✓ Final answer created")
    
    return {
        "final_answer": final_text
    }

# ============================================================================
# STEP 6: Build the Graph
# ============================================================================
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("question_1", get_most_read_article)
workflow.add_node("question_2", recommend_product_based_on_article)
workflow.add_node("synthesize", create_final_answer)

# Define flow: Q1 → Q2 → Synthesize → End
workflow.set_entry_point("question_1")
workflow.add_edge("question_1", "question_2")
workflow.add_edge("question_2", "synthesize")
workflow.add_edge("synthesize", END)

# Compile with memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# ============================================================================
# STEP 7: Helper Functions
# ============================================================================
def fetch_user_reading_history(user_id: str) -> dict:
    """
    Replace this with your actual database query.
    """
    # Example data - replace with:
    # return db.users.find_one({"user_id": user_id})
    
    mock_data = {
        "user-123": {
            "most_frequent_article": "Understanding Cryptocurrency Markets",
            "category": "Finance/Technology",
            "read_count": 15
        },
        "user-456": {
            "most_frequent_article": "Climate Change Solutions",
            "category": "Environment",
            "read_count": 22
        }
    }
    
    return mock_data.get(user_id, {
        "most_frequent_article": "General News Digest",
        "category": "News",
        "read_count": 5
    })

# ============================================================================
# STEP 8: Main Function
# ============================================================================
def process_dependent_query(query: str, user_id: str, thread_id: str = "default"):
    """
    Process a query with dependent questions.
    
    Args:
        query: The full query (e.g., "What is my most read article and based on that recommend a product?")
        user_id: User identifier
        thread_id: Conversation thread identifier
        
    Returns:
        dict with final_answer and intermediate results
    """
    config = {"configurable": {"thread_id": thread_id}}
    
    initial_state = {
        "user_id": user_id,
        "original_query": query,
        "most_read_article": None,
        "recommended_product": None,
        "final_answer": ""
    }
    
    # Run the agent
    result = app.invoke(initial_state, config=config)
    
    return {
        "answer": result["final_answer"],
        "step1_result": result["most_read_article"],
        "step2_result": result["recommended_product"]
    }

# ============================================================================
# STEP 9: Usage Example
# ============================================================================
if __name__ == "__main__":
    # Test the agent
    result = process_dependent_query(
        query="What is my most frequently read news article and based on that which product should you recommend?",
        user_id="user-123",
        thread_id="session-001"
    )
    
    print("\n" + "="*70)
    print("FINAL ANSWER:")
    print("="*70)
    print(result["answer"])
    print("\n" + "="*70)
    print("INTERMEDIATE RESULTS:")
    print("="*70)
    print(f"Step 1 (Most Read Article): {result['step1_result']}")
    print(f"Step 2 (Recommendation): {result['step2_result'][:100]}...")
```

**Key Points:**
- ✅ Simple and clear flow
- ✅ Easy to debug
- ✅ Full control over state
- ✅ Perfect for 2-3 sequential steps

---

### Approach 2: Using Tools (More Flexible)

**Best for:** Dynamic queries where LLM decides the execution flow.

**Architecture:**
```
Query → LLM Agent decides flow → Calls tools in order → Final Answer
```

**Implementation:**
```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated

# ============================================================================
# Define Tools (Each tool = one sub-question)
# ============================================================================

@tool
def get_user_most_read_article(
    user_id: Annotated[str, "The user ID to query"]
) -> str:
    """Get the user's most frequently read news article.
    
    Returns the title, category, and read count.
    """
    # Your database query here
    data = fetch_user_reading_history(user_id)
    
    article = data.get("most_frequent_article", "N/A")
    category = data.get("category", "N/A")
    count = data.get("read_count", 0)
    
    return f"Most read article: '{article}' (Category: {category}, Read {count} times)"

@tool
def recommend_product_for_article(
    article_title: Annotated[str, "The article title to base recommendation on"],
    user_id: Annotated[str, "The user ID for personalization"]
) -> str:
    """Recommend a product based on the article the user reads most.
    
    This tool should be called AFTER getting the most read article.
    Use the article title from the first tool call.
    """
    # Product recommendation logic
    recommendations = {
        "Understanding Cryptocurrency Markets": {
            "product": "Crypto Portfolio Tracker Pro",
            "reason": "Matches your interest in cryptocurrency investments",
            "confidence": 0.92
        },
        "Climate Change Solutions": {
            "product": "Green Energy Home Monitor",
            "reason": "Aligns with your environmental awareness",
            "confidence": 0.88
        }
    }
    
    rec = recommendations.get(
        article_title,
        {
            "product": "Premium News Digest",
            "reason": "General upgrade for news enthusiasts",
            "confidence": 0.65
        }
    )
    
    return f"""Product Recommendation:
- Product: {rec['product']}
- Why: {rec['reason']}
- Confidence: {rec['confidence']}
- Personalized for: {user_id}"""

# ============================================================================
# Create Agent with Tools
# ============================================================================

llm = ChatOpenAI(model="gpt-4", temperature=0)
tools = [get_user_most_read_article, recommend_product_for_article]

agent = create_react_agent(
    llm,
    tools,
    checkpointer=MemorySaver(),
    state_modifier="""You are a helpful assistant that answers multi-step questions.

When asked a question with two parts where the second depends on the first:
1. First, use the appropriate tool to answer the first question
2. Then, use the result from step 1 to answer the second question
3. Provide a comprehensive, natural answer combining both results

Always execute tools in the correct order based on dependencies."""
)

# ============================================================================
# Usage
# ============================================================================

def ask_agent(query: str, user_id: str, thread_id: str = "default"):
    """Ask the agent a dependent question."""
    
    config = {"configurable": {"thread_id": thread_id}}
    
    # Add user context to query
    enhanced_query = f"User ID: {user_id}\n\nQuestion: {query}"
    
    result = agent.invoke(
        {"messages": [("user", enhanced_query)]},
        config=config
    )
    
    return result["messages"][-1].content

# Test
response = ask_agent(
    query="What is my most frequently read article and what product should you recommend based on that?",
    user_id="user-123",
    thread_id="tool-session-001"
)

print(response)

def fetch_user_reading_history(user_id: str) -> dict:
    """Mock database - replace with real DB call"""
    return {
        "most_frequent_article": "Understanding Cryptocurrency Markets",
        "category": "Finance/Technology",
        "read_count": 15
    }
```

**Key Points:**
- ✅ Flexible - LLM decides flow
- ✅ Handles dynamic queries
- ✅ Easy to add new tools
- ❌ Less predictable
- ❌ Higher LLM costs

---

### Approach 3: Async for Better Performance

**Best for:** High-performance production systems.
```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from typing import TypedDict, Optional
import asyncio

# ============================================================================
# State Definition
# ============================================================================
class AsyncAgentState(TypedDict):
    user_id: str
    original_query: str
    step1_result: Optional[str]
    step2_result: Optional[str]
    final_answer: str

llm = ChatOpenAI(model="gpt-4", temperature=0)

# ============================================================================
# Async Node 1: First Question
# ============================================================================
async def async_get_article(state: AsyncAgentState):
    """Async version - better for production."""
    user_id = state["user_id"]
    
    # Simulate async DB call
    reading_data = await async_fetch_reading_history(user_id)
    most_read = reading_data["most_frequent_article"]
    
    return {"step1_result": most_read}

# ============================================================================
# Async Node 2: Second Question (depends on Node 1)
# ============================================================================
async def async_recommend_product(state: AsyncAgentState):
    """Uses result from first question."""
    article = state["step1_result"]  # ← Dependency on step 1
    user_id = state["user_id"]
    
    system_prompt = f"""Based on article: "{article}"
Recommend a product. Be specific and helpful."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Recommend for {user_id}")
    ]
    
    # Async LLM call
    response = await llm.ainvoke(messages)
    
    return {"step2_result": response.content}

# ============================================================================
# Async Node 3: Synthesize
# ============================================================================
async def async_synthesize(state: AsyncAgentState):
    """Create final answer."""
    system_prompt = f"""Combine these into a natural answer:
1. Most read: {state['step1_result']}
2. Recommendation: {state['step2_result']}

Original question: {state['original_query']}"""

    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content="Create answer")
    ])
    
    return {"final_answer": response.content}

# ============================================================================
# Build Async Graph
# ============================================================================
workflow = StateGraph(AsyncAgentState)
workflow.add_node("step1", async_get_article)
workflow.add_node("step2", async_recommend_product)
workflow.add_node("synthesize", async_synthesize)

workflow.set_entry_point("step1")
workflow.add_edge("step1", "step2")
workflow.add_edge("step2", "synthesize")
workflow.add_edge("synthesize", END)

# Use SQLite for persistence
memory = SqliteSaver.from_conn_string("agent_memory.db")
app = workflow.compile(checkpointer=memory)

# ============================================================================
# Async Helper Functions
# ============================================================================
async def async_fetch_reading_history(user_id: str) -> dict:
    """Async database query simulation."""
    # Simulate async DB call
    await asyncio.sleep(0.1)  # Replace with: await db.find_one(...)
    
    return {
        "most_frequent_article": "Understanding Cryptocurrency Markets",
        "category": "Finance/Technology",
        "read_count": 15
    }

# ============================================================================
# Main Async Function
# ============================================================================
async def process_query_async(query: str, user_id: str, thread_id: str = "default"):
    """Process query asynchronously."""
    config = {"configurable": {"thread_id": thread_id}}
    
    initial_state = {
        "user_id": user_id,
        "original_query": query,
        "step1_result": None,
        "step2_result": None,
        "final_answer": ""
    }
    
    result = await app.ainvoke(initial_state, config=config)
    
    return result["final_answer"]

# ============================================================================
# Usage
# ============================================================================
async def main():
    answer = await process_query_async(
        query="What is my most read article and what product should I buy?",
        user_id="user-123",
        thread_id="async-001"
    )
    
    print(answer)

# Run
if __name__ == "__main__":
    asyncio.run(main())
```

**Key Points:**
- ✅ Fast and efficient
- ✅ Production-ready
- ✅ Handles concurrent requests
- ❌ More complex code

---

## Complete Production Example

Full production-ready implementation with error handling, logging, and validation.
```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from typing import TypedDict, Optional
import logging
from datetime import datetime

# ============================================================================
# Setup Logging
# ============================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# State with Full Tracking
# ============================================================================
class ProductionState(TypedDict):
    user_id: str
    original_query: str
    timestamp: str
    
    # Step results
    most_read_article: Optional[str]
    article_category: Optional[str]
    read_count: Optional[int]
    recommended_product: Optional[str]
    
    # Final output
    final_answer: str
    
    # Error handling
    error: Optional[str]

llm = ChatOpenAI(model="gpt-4", temperature=0)

# ============================================================================
# Node 1: Get Article with Error Handling
# ============================================================================
def get_article_safe(state: ProductionState):
    """Question 1 with error handling."""
    try:
        user_id = state["user_id"]
        logger.info(f"Fetching article for user: {user_id}")
        
        # Your actual database call
        data = fetch_from_database(user_id)
        
        if not data:
            return {
                "error": f"No reading history found for user {user_id}",
                "final_answer": "I couldn't find any reading history for your account."
            }
        
        logger.info(f"✓ Found article: {data['most_frequent_article']}")
        
        return {
            "most_read_article": data["most_frequent_article"],
            "article_category": data["category"],
            "read_count": data["read_count"]
        }
        
    except Exception as e:
        logger.error(f"Error in get_article: {e}")
        return {
            "error": str(e),
            "final_answer": "Sorry, I encountered an error fetching your reading history."
        }

# ============================================================================
# Node 2: Recommend Product with Validation
# ============================================================================
def recommend_product_safe(state: ProductionState):
    """Question 2 with validation."""
    
    # Check if step 1 had errors
    if state.get("error"):
        return {}  # Skip this step
    
    try:
        article = state["most_read_article"]
        category = state["article_category"]
        user_id = state["user_id"]
        
        logger.info(f"Generating recommendation based on: {article}")
        
        system_prompt = f"""You are a product recommendation expert.

User Profile:
- Most read article: "{article}"
- Category: {category}
- User ID: {user_id}

Recommend ONE product that:
1. Relates to their reading interest
2. Provides clear value
3. Is specific and actionable

Return JSON with: product_name, reason, benefit, confidence_score"""

        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content="Provide recommendation")
        ])
        
        logger.info(f"✓ Generated recommendation")
        
        return {
            "recommended_product": response.content
        }
        
    except Exception as e:
        logger.error(f"Error in recommend_product: {e}")
        return {
            "error": str(e),
            "recommended_product": "Unable to generate recommendation at this time."
        }

# ============================================================================
# Node 3: Create Final Answer
# ============================================================================
def create_answer_safe(state: ProductionState):
    """Synthesize with error handling."""
    
    if state.get("error"):
        # Already have error message
        return {}
    
    try:
        system_prompt = f"""Create a friendly, helpful answer.

Original question: {state['original_query']}

Information:
- Most read: {state['most_read_article']} ({state['read_count']} times)
- Category: {state['article_category']}
- Recommendation: {state['recommended_product']}

Write a natural, conversational response."""

        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content="Create final answer")
        ])
        
        return {"final_answer": response.content}
        
    except Exception as e:
        logger.error(f"Error in synthesize: {e}")
        return {
            "final_answer": "I found your information but had trouble formatting the response. Please try again."
        }

# ============================================================================
# Build Production Graph
# ============================================================================
workflow = StateGraph(ProductionState)

workflow.add_node("get_article", get_article_safe)
workflow.add_node("recommend", recommend_product_safe)
workflow.add_node("synthesize", create_answer_safe)

workflow.set_entry_point("get_article")
workflow.add_edge("get_article", "recommend")
workflow.add_edge("recommend", "synthesize")
workflow.add_edge("synthesize", END)

# Persistent storage
memory = SqliteSaver.from_conn_string("production_agent.db")
app = workflow.compile(checkpointer=memory)

# ============================================================================
# Database Function (Replace with your actual DB)
# ============================================================================
def fetch_from_database(user_id: str) -> dict:
    """
    Replace this with your actual database query.
    
    Example with MongoDB:
        from motor.motor_asyncio import AsyncIOMotorClient
        client = AsyncIOMotorClient(MONGO_URI)
        db = client["your_database"]
        collection = db["user_reading_history"]
        return collection.find_one({"user_id": user_id})
    
    Example with PostgreSQL:
        import psycopg2
        conn = psycopg2.connect(...)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM reading_history WHERE user_id = %s", (user_id,))
        return cursor.fetchone()
    """
    # Mock data - REPLACE THIS
    mock_db = {
        "user-123": {
            "most_frequent_article": "Understanding Cryptocurrency Markets",
            "category": "Finance/Technology",
            "read_count": 15
        }
    }
    
    return mock_db.get(user_id)

# ============================================================================
# Main API Function
# ============================================================================
def ask_agent(query: str, user_id: str, thread_id: str = None) -> dict:
    """
    Main function to call from your API/application.
    
    Args:
        query: User's question
        user_id: User identifier
        thread_id: Optional conversation thread ID
        
    Returns:
        dict with answer and metadata
    """
    if thread_id is None:
        import uuid
        thread_id = str(uuid.uuid4())
    
    config = {"configurable": {"thread_id": thread_id}}
    
    initial_state = {
        "user_id": user_id,
        "original_query": query,
        "timestamp": datetime.utcnow().isoformat(),
        "most_read_article": None,
        "article_category": None,
        "read_count": None,
        "recommended_product": None,
        "final_answer": "",
        "error": None
    }
    
    try:
        result = app.invoke(initial_state, config=config)
        
        return {
            "success": result.get("error") is None,
            "answer": result["final_answer"],
            "metadata": {
                "article": result.get("most_read_article"),
                "category": result.get("article_category"),
                "thread_id": thread_id,
                "timestamp": result["timestamp"]
            }
        }
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return {
            "success": False,
            "answer": "I apologize, but I encountered a technical error. Please try again.",
            "error": str(e)
        }

# ============================================================================
# Usage Examples
# ============================================================================
if __name__ == "__main__":
    # Example 1: Basic usage
    result = ask_agent(
        query="What is my most read article and what product should you recommend?",
        user_id="user-123"
    )
    
    print("Answer:", result["answer"])
    print("Success:", result["success"])
    
    # Example 2: With specific thread
    result2 = ask_agent(
        query="Based on my reading, what should I buy?",
        user_id="user-123",
        thread_id="session-abc-123"
    )
    
    print("\nAnswer:", result2["answer"])
```

---

## Testing Your Implementation
```python
import pytest

def test_sequential_flow():
    """Test that step 2 receives step 1 result."""
    result = ask_agent(
        query="What is my most read article and recommend a product?",
        user_id="user-123"
    )
    
    assert result["success"] == True
    assert len(result["answer"]) > 0
    assert result["metadata"]["article"] is not None

def test_error_handling():
    """Test error handling for non-existent user."""
    result = ask_agent(
        query="What should I read?",
        user_id="nonexistent-user"
    )
    
    # Should still return a response, not crash
    assert "answer" in result

def test_dependency():
    """Test that Q2 actually uses Q1 answer."""
    # Mock the state at step 2
    state = {
        "user_id": "test",
        "most_read_article": "Test Article",
        "article_category": "Test",
        "original_query": "test query"
    }
    
    result = recommend_product_safe(state)
    
    # Check that the article name appears in recommendation
    assert "Test Article" in str(result)

def test_full_integration():
    """Integration test of entire flow."""
    result = process_dependent_query(
        query="What is my most read article and what should I buy?",
        user_id="user-123",
        thread_id="test-integration"
    )
    
    assert "answer" in result
    assert "step1_result" in result
    assert "step2_result" in result
    assert result["step1_result"] is not None
```

**Run Tests:**
```bash
pytest test_agent.py -v
```

---

## Quick Comparison

| Approach | Best For | Complexity | Flexibility | Performance | Cost | Debuggability |
|----------|----------|------------|-------------|-------------|------|---------------|
| **Sequential State** | Fixed 2-3 steps | ⭐ Low | ⭐⭐ Medium | ⚡⚡⚡ Fast | 💰 Low | ✅✅✅ Easy |
| **Tools** | Dynamic queries | ⭐⭐ Medium | ⭐⭐⭐ High | ⚡⚡ Medium | 💰💰 Medium | ✅✅ Medium |
| **Async** | Production/Scale | ⭐⭐⭐ High | ⭐⭐ Medium | ⚡⚡⚡ Fast | 💰 Low | ✅✅ Medium |

### When to Use Each:

**✅ Use Sequential State (Approach 1) when:**
- You have 2-3 clear sequential steps
- Flow is predictable and fixed
- You need easy debugging
- Budget is limited
- **← RECOMMENDED FOR YOUR USE CASE**

**✅ Use Tools (Approach 2) when:**
- Questions are dynamic
- LLM should decide the flow
- You need maximum flexibility
- You have well-defined tools

**✅ Use Async (Approach 3) when:**
- You need high performance
- Handling many concurrent requests
- Building production system
- Database calls are slow

---

## Key Takeaways

### ✅ Critical Requirements

1. **Store intermediate results in state**
```python
   # Step 1 stores result
   return {"most_read_article": article}
   
   # Step 2 accesses it
   article = state["most_read_article"]
```

2. **Use proper state typing**
```python
   class AgentState(TypedDict):
       step1_result: Optional[str]  # Result from step 1
       step2_result: Optional[str]  # Result from step 2
```

3. **Define clear edges**
```python
   workflow.add_edge("step1", "step2")  # step2 runs after step1
   workflow.add_edge("step2", "step3")  # step3 runs after step2
```

4. **Add error handling**
```python
   if not state.get("step1_result"):
       return {"error": "Step 1 failed"}
```

5. **Use persistent checkpointing for production**
```python
   memory = SqliteSaver.from_conn_string("agent.db")
   app = workflow.compile(checkpointer=memory)
```

---

## FAQ

### Q: How do I pass data between steps?

**A:** Store it in the state:
```python
# Step 1
def step1(state):
    result = "my data"
    return {"step1_data": result}

# Step 2
def step2(state):
    previous_data = state["step1_data"]  # Access step 1's data
    # Use it...
```

### Q: What if Step 1 fails?

**A:** Add error handling:
```python
def step1(state):
    try:
        result = risky_operation()
        return {"step1_result": result}
    except Exception as e:
        return {"error": str(e)}

def step2(state):
    if state.get("error"):
        return {}  # Skip this step
    # Normal processing...
```

### Q: Can I have more than 2 dependent steps?

**A:** Yes! Just add more nodes:
```python
workflow.add_node("step1", step1_func)
workflow.add_node("step2", step2_func)
workflow.add_node("step3", step3_func)
workflow.add_node("step4", step4_func)

workflow.add_edge("step1", "step2")
workflow.add_edge("step2", "step3")
workflow.add_edge("step3", "step4")
```

### Q: How do I debug which step is failing?

**A:** Add logging:
```python
import logging

logger = logging.getLogger(__name__)

def step1(state):
    logger.info(f"Starting step1 with state: {state}")
    result = process()
    logger.info(f"Step1 result: {result}")
    return {"step1_result": result}
```

### Q: Can I use this with FastAPI?

**A:** Yes!
```python
from fastapi import FastAPI

app_api = FastAPI()

@app_api.post("/ask")
async def ask_endpoint(query: str, user_id: str):
    result = ask_agent(query, user_id)
    return result
```

### Q: How do I visualize the graph?

**A:** Use the built-in visualization:
```python
from IPython.display import Image

# Generate graph visualization
Image(app.get_graph().draw_mermaid_png())
```

---

## Additional Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangGraph Examples](https://github.com/langchain-ai/langgraph/tree/main/examples)
- [State Management Guide](https://langchain-ai.github.io/langgraph/concepts/low_level/)
- [Checkpointing Guide](https://langchain-ai.github.io/langgraph/concepts/persistence/)

---

## Contributing

Found an issue or want to improve this guide? Please:
1. Open an issue
2. Submit a pull request
3. Share your feedback

---

## License

This guide is provided as-is for educational purposes.

---

**Last Updated:** December 2024

**Version:** 1.0

**Author:** LangGraph Community
