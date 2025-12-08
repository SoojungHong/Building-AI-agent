# LangChain & LangGraph Memory Guide

Complete guide to implementing memory in LangChain and LangGraph agents with code examples.

## Table of Contents
- [Introduction](#introduction)
- [LangChain Memory (Traditional)](#langchain-memory-traditional)
  - [Conversation Buffer Memory](#conversation-buffer-memory)
  - [Conversation Buffer Window Memory](#conversation-buffer-window-memory)
  - [Conversation Summary Memory](#conversation-summary-memory)
- [LangGraph Memory (Modern - Recommended)](#langgraph-memory-modern---recommended)
  - [Basic Memory with MemorySaver](#basic-memory-with-memorysaver)
  - [Persistent Memory with SQLite](#persistent-memory-with-sqlite)
  - [Production Memory with PostgreSQL](#production-memory-with-postgresql)
- [Complete Agent Examples](#complete-agent-examples)
- [Custom State Management](#custom-state-management)
- [Async Memory Usage](#async-memory-usage)
- [Memory Management Functions](#memory-management-functions)
- [Production Example with MongoDB](#production-example-with-mongodb)
- [Best Practices](#best-practices)

---

## Introduction

Memory allows your AI agents to remember past conversations and maintain context across multiple interactions. This guide covers both LangChain's traditional memory approaches and LangGraph's modern checkpointing system.

**Quick Recommendation**: Use **LangGraph with checkpointing** for new projects. It's more powerful, flexible, and scales better.

---

## LangChain Memory (Traditional)

### Conversation Buffer Memory

Stores all conversation history in memory.
```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI

# Initialize memory
memory = ConversationBufferMemory()

# Create LLM
llm = ChatOpenAI(temperature=0.7, model="gpt-4")

# Create conversation chain with memory
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True  # See what's happening
)

# Use it
response1 = conversation.predict(input="Hi, my name is Alice")
print(response1)

response2 = conversation.predict(input="What's my name?")
print(response2)  # Should remember "Alice"

# View memory
print(memory.load_memory_variables({}))
```

**Pros:**
- Simple to implement
- Stores complete conversation history

**Cons:**
- Memory grows infinitely
- Not suitable for long conversations

---

### Conversation Buffer Window Memory

Keeps only the last N interactions.
```python
from langchain.memory import ConversationBufferWindowMemory

# Keep only last 5 interactions
memory = ConversationBufferWindowMemory(k=5)

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)
```

**Pros:**
- Limits memory usage
- Good for long-running conversations

**Cons:**
- Older context is lost
- May lose important information

---

### Conversation Summary Memory

Automatically summarizes conversation history.
```python
from langchain.memory import ConversationSummaryMemory

# Automatically summarizes long conversations
memory = ConversationSummaryMemory(llm=llm)

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)
```

**Pros:**
- Efficient for long conversations
- Retains key information

**Cons:**
- Requires additional LLM calls
- Summary quality depends on LLM

---

## LangGraph Memory (Modern - Recommended)

LangGraph uses **checkpointing** for memory management. This approach is more powerful and flexible than traditional LangChain memory.

### Basic Memory with MemorySaver

In-memory storage, perfect for development and testing.
```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated
from operator import add

# Define state
class AgentState(TypedDict):
    messages: Annotated[list, add]  # Accumulate messages
    user_info: dict  # Store user information

# Initialize LLM
llm = ChatOpenAI(model="gpt-4")

# Define agent node
def agent_node(state: AgentState):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

# Create graph
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.set_entry_point("agent")
workflow.add_edge("agent", END)

# **KEY: Add memory with checkpointer**
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Use with thread_id for conversation tracking
config = {"configurable": {"thread_id": "conversation-1"}}

# First message
response1 = app.invoke(
    {"messages": [("user", "Hi, my name is Bob")]},
    config=config
)
print(response1["messages"][-1].content)

# Second message - remembers context!
response2 = app.invoke(
    {"messages": [("user", "What's my name?")]},
    config=config
)
print(response2["messages"][-1].content)  # Should say "Bob"
```

**Pros:**
- Simple setup
- Fast (in-memory)
- Great for development

**Cons:**
- Lost on restart
- Not suitable for production

---

### Persistent Memory with SQLite

File-based storage for persistent conversations.
```python
from langgraph.checkpoint.sqlite import SqliteSaver

# Use SQLite for persistent storage
memory = SqliteSaver.from_conn_string("checkpoints.db")

app = workflow.compile(checkpointer=memory)

# Same usage as before
config = {"configurable": {"thread_id": "user-123"}}
response = app.invoke({"messages": [("user", "Hello")]}, config=config)
```

**Pros:**
- Persistent across restarts
- No external database needed
- Good for small-scale production

**Cons:**
- Single file limitation
- Not suitable for high concurrency

---

### Production Memory with PostgreSQL

Scalable database storage for production environments.
```python
from langgraph.checkpoint.postgres import PostgresSaver

# Production-ready persistent memory
connection_string = "postgresql://user:password@localhost:5432/langraph_db"
memory = PostgresSaver.from_conn_string(connection_string)

app = workflow.compile(checkpointer=memory)

# Use exactly like MemorySaver
config = {"configurable": {"thread_id": "user-456"}}
response = app.invoke({"messages": [("user", "Hello")]}, config=config)
```

**Pros:**
- Production-ready
- Handles high concurrency
- Scalable

**Cons:**
- Requires PostgreSQL setup
- More complex infrastructure

---

## Complete Agent Examples

### Full-Featured Agent with Memory
```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import TypedDict, Annotated, Sequence
from operator import add

# Define state with message accumulation
class AgentState(TypedDict):
    messages: Annotated[Sequence, add]

# Initialize components
llm = ChatOpenAI(model="gpt-4", temperature=0.7)
memory = MemorySaver()

# System prompt
SYSTEM_PROMPT = """You are a helpful financial advisor assistant. 
Remember user information and provide personalized advice based on conversation history."""

# Agent function
def agent(state: AgentState):
    messages = state["messages"]
    
    # Add system message if first interaction
    if len(messages) == 1:
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    
    response = llm.invoke(messages)
    return {"messages": [response]}

# Build graph
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent)
workflow.set_entry_point("agent")
workflow.add_edge("agent", END)

# Compile with memory
app = workflow.compile(checkpointer=memory)

# Helper function to chat
def chat(user_input: str, thread_id: str = "default"):
    config = {"configurable": {"thread_id": thread_id}}
    
    result = app.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=config
    )
    
    return result["messages"][-1].content

# Test conversation
print(chat("Hi, I'm interested in investing in tech stocks.", "user-001"))
print(chat("What should I know about risk?", "user-001"))
print(chat("What did I say I was interested in?", "user-001"))  # Remembers!
```

---

## Custom State Management

Track additional information beyond just messages.
```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict

class CustomAgentState(TypedDict):
    messages: list
    user_profile: dict
    conversation_count: int
    last_topic: str

def agent_with_tracking(state: CustomAgentState):
    messages = state["messages"]
    user_profile = state.get("user_profile", {})
    
    # Use profile in prompt
    context = f"User info: {user_profile}" if user_profile else ""
    
    # Create enhanced prompt with context
    enhanced_messages = messages.copy()
    if context and len(messages) > 0:
        enhanced_messages[0] = HumanMessage(
            content=f"{context}\n\n{messages[0].content}"
        )
    
    response = llm.invoke(enhanced_messages)
    
    return {
        "messages": [response],
        "conversation_count": state.get("conversation_count", 0) + 1,
        "last_topic": "extracted_topic_here"  # Extract from conversation
    }

# Build and compile
workflow = StateGraph(CustomAgentState)
workflow.add_node("agent", agent_with_tracking)
workflow.set_entry_point("agent")
workflow.add_edge("agent", END)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Initialize with user profile
config = {"configurable": {"thread_id": "user-123"}}
initial_state = {
    "messages": [HumanMessage(content="Hello")],
    "user_profile": {"name": "Alice", "risk_tolerance": "moderate"},
    "conversation_count": 0,
    "last_topic": ""
}

response = app.invoke(initial_state, config=config)
print(f"Conversation count: {response['conversation_count']}")
```

---

## Async Memory Usage

For high-performance applications using async/await.
```python
import asyncio
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage

# Async agent
async def async_agent(state: AgentState):
    messages = state["messages"]
    response = await llm.ainvoke(messages)
    return {"messages": [response]}

# Build async workflow
workflow = StateGraph(AgentState)
workflow.add_node("agent", async_agent)
workflow.set_entry_point("agent")
workflow.add_edge("agent", END)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Async chat function
async def chat_async(user_input: str, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    result = await app.ainvoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=config
    )
    return result["messages"][-1].content

# Run async
async def main():
    response1 = await chat_async("Hello!", "user-001")
    print(response1)
    
    response2 = await chat_async("What did I just say?", "user-001")
    print(response2)

# Execute
asyncio.run(main())
```

---

## Memory Management Functions

Utilities for managing conversation history.
```python
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Get conversation history
def get_history(thread_id: str):
    """Retrieve full conversation history for a thread."""
    config = {"configurable": {"thread_id": thread_id}}
    state = app.get_state(config)
    return state.values.get("messages", [])

# Clear conversation
def clear_history(thread_id: str):
    """Clear all messages for a specific thread."""
    config = {"configurable": {"thread_id": thread_id}}
    app.update_state(config, {"messages": []})

# Get conversation summary
def get_conversation_summary(thread_id: str):
    """Get a summary of the conversation."""
    messages = get_history(thread_id)
    return {
        "thread_id": thread_id,
        "message_count": len(messages),
        "first_message": messages[0].content if messages else None,
        "last_message": messages[-1].content if messages else None
    }

# Usage examples
print(get_history("user-001"))
print(get_conversation_summary("user-001"))
clear_history("user-001")
```

---

## Production Example with MongoDB

Custom memory implementation using MongoDB for full control.
```python
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime
from typing import Dict, Optional
import json

class MongoMemory:
    """Custom memory implementation using MongoDB."""
    
    def __init__(self, mongo_uri: str, db_name: str):
        self.client = AsyncIOMotorClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db["conversation_memory"]
        
    async def initialize(self):
        """Create indexes for better performance."""
        await self.collection.create_index("thread_id", unique=True)
        await self.collection.create_index("updated_at")
    
    async def save_state(self, thread_id: str, state: dict):
        """Save conversation state to MongoDB."""
        await self.collection.update_one(
            {"thread_id": thread_id},
            {
                "$set": {
                    "state": state,
                    "updated_at": datetime.utcnow(),
                    "message_count": len(state.get("messages", []))
                }
            },
            upsert=True
        )
    
    async def load_state(self, thread_id: str) -> Optional[Dict]:
        """Load conversation state from MongoDB."""
        doc = await self.collection.find_one({"thread_id": thread_id})
        return doc["state"] if doc else None
    
    async def delete_state(self, thread_id: str):
        """Delete conversation state."""
        await self.collection.delete_one({"thread_id": thread_id})
    
    async def list_threads(self, limit: int = 100):
        """List all conversation threads."""
        cursor = self.collection.find().sort("updated_at", -1).limit(limit)
        return await cursor.to_list(length=limit)
    
    async def get_stats(self, thread_id: str):
        """Get conversation statistics."""
        doc = await self.collection.find_one({"thread_id": thread_id})
        if not doc:
            return None
        
        return {
            "thread_id": thread_id,
            "message_count": doc.get("message_count", 0),
            "created_at": doc.get("created_at"),
            "updated_at": doc.get("updated_at"),
            "last_activity": (datetime.utcnow() - doc["updated_at"]).total_seconds()
        }

# Usage in agent
mongo_memory = MongoMemory("mongodb://localhost:27017", "langraph_db")

async def agent_with_mongo(state: AgentState, thread_id: str):
    """Agent with MongoDB memory integration."""
    
    # Load previous state
    previous_state = await mongo_memory.load_state(thread_id)
    if previous_state:
        # Merge previous messages with new ones
        state["messages"] = previous_state.get("messages", []) + state["messages"]
    
    # Process with LLM
    response = await llm.ainvoke(state["messages"])
    state["messages"].append(response)
    
    # Save updated state
    await mongo_memory.save_state(thread_id, state)
    
    return state

# Initialize and use
async def main():
    await mongo_memory.initialize()
    
    # Use the agent
    state = {"messages": [HumanMessage(content="Hello!")]}
    result = await agent_with_mongo(state, "user-001")
    
    # Get stats
    stats = await mongo_memory.get_stats("user-001")
    print(stats)

asyncio.run(main())
```

---

## Best Practices

### 1. Choose the Right Memory Type

| Use Case | Recommended Solution |
|----------|---------------------|
| Development/Testing | `MemorySaver` (in-memory) |
| Small Production | `SqliteSaver` |
| Large Production | `PostgresSaver` |
| Custom Requirements | Custom implementation (MongoDB, Redis, etc.) |

### 2. Thread ID Strategy
```python
# User-based threads
thread_id = f"user-{user_id}"

# Session-based threads
thread_id = f"session-{session_id}"

# Conversation-based threads
thread_id = f"conversation-{conversation_id}"

# Date-based threads (for daily resets)
from datetime import date
thread_id = f"user-{user_id}-{date.today()}"
```

### 3. Memory Cleanup
```python
from datetime import datetime, timedelta

async def cleanup_old_conversations(days: int = 30):
    """Remove conversations older than N days."""
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    
    result = await mongo_memory.collection.delete_many({
        "updated_at": {"$lt": cutoff_date}
    })
    
    print(f"Deleted {result.deleted_count} old conversations")

# Run periodically
asyncio.run(cleanup_old_conversations(30))
```

### 4. Error Handling
```python
async def safe_chat(user_input: str, thread_id: str):
    """Chat with error handling and fallback."""
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        result = await app.ainvoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config
        )
        return result["messages"][-1].content
    
    except Exception as e:
        print(f"Error in conversation {thread_id}: {e}")
        
        # Fallback: create new conversation
        new_thread_id = f"{thread_id}-{datetime.now().timestamp()}"
        config = {"configurable": {"thread_id": new_thread_id}}
        
        result = await app.ainvoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config
        )
        return result["messages"][-1].content
```

### 5. Monitor Memory Usage
```python
async def get_memory_stats():
    """Get overall memory statistics."""
    stats = await mongo_memory.collection.aggregate([
        {
            "$group": {
                "_id": None,
                "total_conversations": {"$sum": 1},
                "total_messages": {"$sum": "$message_count"},
                "avg_messages": {"$avg": "$message_count"}
            }
        }
    ]).to_list(length=1)
    
    return stats[0] if stats else {}

# Monitor
stats = asyncio.run(get_memory_stats())
print(f"Total conversations: {stats['total_conversations']}")
print(f"Average messages per conversation: {stats['avg_messages']:.2f}")
```

### 6. State Management Best Practices
```python
class AgentState(TypedDict):
    # Always accumulate messages
    messages: Annotated[list, add]
    
    # Other fields should have defaults
    user_profile: dict  # Default: {}
    metadata: dict      # Default: {}
    
    # Use Optional for nullable fields
    current_task: Optional[str]  # Default: None

# Initialize with sensible defaults
def create_initial_state(user_id: str) -> AgentState:
    return {
        "messages": [],
        "user_profile": {},
        "metadata": {
            "user_id": user_id,
            "created_at": datetime.utcnow().isoformat()
        },
        "current_task": None
    }
```

---

## Comparison Table

| Feature | LangChain Memory | LangGraph Memory |
|---------|------------------|------------------|
| **Setup Complexity** | Simple | Moderate |
| **Flexibility** | Limited | High |
| **Persistence** | Requires custom code | Built-in options |
| **State Management** | Basic | Advanced |
| **Scalability** | Limited | Excellent |
| **Async Support** | Limited | Full |
| **Production Ready** | For simple cases | Yes |
| **Recommendation** | Legacy projects | New projects |

---

## Quick Start Templates

### Template 1: Simple Chatbot
```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from typing import TypedDict, Annotated
from operator import add

class State(TypedDict):
    messages: Annotated[list, add]

llm = ChatOpenAI(model="gpt-4")

def chat_node(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

workflow = StateGraph(State)
workflow.add_node("chat", chat_node)
workflow.set_entry_point("chat")
workflow.add_edge("chat", END)

app = workflow.compile(checkpointer=MemorySaver())

def chat(msg: str, thread: str = "default"):
    result = app.invoke(
        {"messages": [HumanMessage(content=msg)]},
        {"configurable": {"thread_id": thread}}
    )
    return result["messages"][-1].content
```

### Template 2: Agent with Tools
```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

llm = ChatOpenAI(model="gpt-4")
tools = [search]

app = create_react_agent(
    llm,
    tools,
    checkpointer=MemorySaver()
)

def chat(msg: str, thread: str = "default"):
    result = app.invoke(
        {"messages": [("user", msg)]},
        {"configurable": {"thread_id": thread}}
    )
    return result["messages"][-1].content
```

---

## Troubleshooting

### Issue: Memory not persisting

**Solution:** Check that you're using the same `thread_id`:
```python
# ❌ Wrong - creates new thread each time
app.invoke({"messages": [...]}, {"configurable": {"thread_id": random_id()}})

# ✅ Correct - reuses same thread
user_thread = f"user-{user_id}"
app.invoke({"messages": [...]}, {"configurable": {"thread_id": user_thread}})
```

### Issue: Memory growing too large

**Solution:** Implement message trimming:
```python
def trim_messages(messages: list, max_messages: int = 20):
    """Keep only the last N messages."""
    if len(messages) > max_messages:
        # Keep system message if present
        system_msg = [m for m in messages if m.type == "system"]
        recent_msgs = messages[-max_messages:]
        return system_msg + recent_msgs
    return messages

def agent_with_trimming(state: AgentState):
    messages = trim_messages(state["messages"])
    response = llm.invoke(messages)
    return {"messages": [response]}
```

### Issue: Async/await errors

**Solution:** Ensure all functions in the chain are async:
```python
# ❌ Wrong - mixing sync and async
async def async_agent(state):
    response = llm.invoke(state["messages"])  # Sync call
    return {"messages": [response]}

# ✅ Correct - all async
async def async_agent(state):
    response = await llm.ainvoke(state["messages"])  # Async call
    return {"messages": [response]}
```

---

## Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Documentation](https://python.langchain.com/docs/)
- [LangGraph Checkpoint Documentation](https://langchain-ai.github.io/langgraph/concepts/persistence/)
- [Example Projects](https://github.com/langchain-ai/langgraph/tree/main/examples)

---

## License

This guide is provided as-is for educational purposes.

---

## Contributing

Feel free to submit issues or pull requests to improve this guide.

---

**Last Updated:** December 2024

