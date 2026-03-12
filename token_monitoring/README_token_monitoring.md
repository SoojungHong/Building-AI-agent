# LangGraph & LangChain: Token Monitoring & LangSmith Tracking Guide

## Table of Contents
1. [Overview](#overview)
2. [Installation & Setup](#installation--setup)
3. [Token Monitoring Methods](#token-monitoring-methods)
   - [Method 1: Callback Handler](#method-1-callback-handler)
   - [Method 2: Response Metadata](#method-2-response-metadata)
   - [Method 3: Custom Token Counter Middleware](#method-3-custom-token-counter-middleware)
   - [Method 4: LangGraph Node-Level Tracking](#method-4-langgraph-node-level-tracking)
4. [LangSmith Integration](#langsmith-integration)
   - [Basic LangSmith Setup](#basic-langsmith-setup)
   - [Custom Tracing with LangSmith](#custom-tracing-with-langsmith)
   - [LangSmith Datasets & Evaluation](#langsmith-datasets--evaluation)
   - [LangSmith Dashboard Metrics](#langsmith-dashboard-metrics)
5. [Full Agent Example with Everything Combined](#full-agent-example-with-everything-combined)
6. [Best Practices](#best-practices)

---

## Overview

This guide covers how to **monitor token usage** and **track agent runs** when building agents with LangGraph and LangChain. Token monitoring helps you manage costs and debug performance, while LangSmith provides deep observability into every step your agent takes.

---

## Installation & Setup

```bash
pip install langgraph langchain langchain-openai langsmith python-dotenv
```

Create a `.env` file:

```env
# OpenAI
OPENAI_API_KEY=your_openai_api_key

# LangSmith
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=my-agent-project   # Optional: name your project
```

---

## Token Monitoring Methods

### Method 1: Callback Handler

Use a custom `BaseCallbackHandler` to intercept LLM calls and capture token usage in real time.

```python
# token_callback.py
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from typing import Any, Dict, List
import time


class TokenUsageCallbackHandler(BaseCallbackHandler):
    """
    A callback handler that tracks token usage across all LLM calls.
    Attach this to any chain or agent to get per-call and cumulative stats.
    """

    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.successful_requests = 0
        self.call_history: List[Dict] = []
        self._start_time: float = 0

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs):
        """Called when LLM starts generating."""
        self._start_time = time.time()
        print(f"\n🚀 LLM Call started | Prompt count: {len(prompts)}")

    def on_llm_end(self, response: LLMResult, **kwargs):
        """Called when LLM finishes. Extract token usage from response metadata."""
        elapsed = time.time() - self._start_time

        # Extract token usage from LLM output metadata
        if response.llm_output and "token_usage" in response.llm_output:
            usage = response.llm_output["token_usage"]
            prompt_tokens     = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens      = usage.get("total_tokens", 0)
        else:
            # Fallback: try response metadata on individual generations
            prompt_tokens, completion_tokens, total_tokens = 0, 0, 0
            for gen_list in response.generations:
                for gen in gen_list:
                    meta = getattr(gen, "generation_info", {}) or {}
                    prompt_tokens     += meta.get("prompt_tokens", 0)
                    completion_tokens += meta.get("completion_tokens", 0)
                    total_tokens      += meta.get("total_tokens", 0)

        # Accumulate totals
        self.prompt_tokens     += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens      += total_tokens
        self.successful_requests += 1

        # Record per-call history
        record = {
            "call_number":        self.successful_requests,
            "prompt_tokens":      prompt_tokens,
            "completion_tokens":  completion_tokens,
            "total_tokens":       total_tokens,
            "elapsed_seconds":    round(elapsed, 3),
        }
        self.call_history.append(record)

        print(f"✅ LLM Call #{self.successful_requests} complete")
        print(f"   Prompt tokens:     {prompt_tokens}")
        print(f"   Completion tokens: {completion_tokens}")
        print(f"   Total tokens:      {total_tokens}")
        print(f"   Time elapsed:      {elapsed:.3f}s")

    def on_llm_error(self, error: Exception, **kwargs):
        print(f"❌ LLM Error: {error}")

    def get_summary(self) -> Dict:
        """Return a cumulative summary across all calls."""
        return {
            "total_prompt_tokens":     self.prompt_tokens,
            "total_completion_tokens": self.completion_tokens,
            "total_tokens":            self.total_tokens,
            "successful_requests":     self.successful_requests,
            "call_history":            self.call_history,
        }

    def reset(self):
        """Reset all counters."""
        self.__init__()


# ── Usage ──────────────────────────────────────────────────────────────────────

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

tracker = TokenUsageCallbackHandler()
llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[tracker])

response = llm.invoke([HumanMessage(content="What is the capital of France?")])
print("\n📊 Token Summary:", tracker.get_summary())
```

---

### Method 2: Response Metadata

For a lightweight, one-off approach, read token usage directly from the response object — no callback needed.

```python
# response_metadata.py
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


def invoke_with_token_info(llm: ChatOpenAI, prompt: str) -> dict:
    """Invoke LLM and return both the reply and token usage."""
    response = llm.invoke([HumanMessage(content=prompt)])

    usage = response.response_metadata.get("token_usage", {})
    return {
        "content": response.content,
        "prompt_tokens":     usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens":      usage.get("total_tokens", 0),
        "model":             response.response_metadata.get("model_name", "unknown"),
    }


llm = ChatOpenAI(model="gpt-4o-mini")
result = invoke_with_token_info(llm, "Explain quantum computing in one sentence.")

print(f"Response:          {result['content']}")
print(f"Prompt tokens:     {result['prompt_tokens']}")
print(f"Completion tokens: {result['completion_tokens']}")
print(f"Total tokens:      {result['total_tokens']}")
print(f"Model:             {result['model']}")
```

---

### Method 3: Custom Token Counter Middleware

Wrap your LLM in a reusable class that accumulates stats and enforces a token budget.

```python
# token_counter_middleware.py
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from typing import List, Optional
from dataclasses import dataclass, field
import time


@dataclass
class TokenStats:
    prompt_tokens: int     = 0
    completion_tokens: int = 0
    total_tokens: int      = 0
    call_count: int        = 0
    total_cost_usd: float  = 0.0

    # gpt-4o-mini pricing (per 1 000 tokens, as of early 2025)
    COST_PER_1K_INPUT:  float = 0.00015
    COST_PER_1K_OUTPUT: float = 0.00060

    def add(self, prompt: int, completion: int):
        self.prompt_tokens     += prompt
        self.completion_tokens += completion
        self.total_tokens      += prompt + completion
        self.call_count        += 1
        self.total_cost_usd    += (
            (prompt     / 1000) * self.COST_PER_1K_INPUT +
            (completion / 1000) * self.COST_PER_1K_OUTPUT
        )

    def report(self) -> str:
        return (
            f"Calls: {self.call_count} | "
            f"Prompt: {self.prompt_tokens} | "
            f"Completion: {self.completion_tokens} | "
            f"Total: {self.total_tokens} | "
            f"Est. cost: ${self.total_cost_usd:.6f}"
        )


class TokenAwareLLM:
    """Thin wrapper around ChatOpenAI with built-in token tracking and budget guard."""

    def __init__(self, model: str = "gpt-4o-mini", token_budget: Optional[int] = None):
        self.llm = ChatOpenAI(model=model)
        self.stats = TokenStats()
        self.token_budget = token_budget

    def invoke(self, messages: List[BaseMessage]) -> str:
        if self.token_budget and self.stats.total_tokens >= self.token_budget:
            raise RuntimeError(
                f"Token budget exceeded: {self.stats.total_tokens} / {self.token_budget}"
            )

        start = time.time()
        response = self.llm.invoke(messages)
        elapsed = time.time() - start

        usage = response.response_metadata.get("token_usage", {})
        self.stats.add(
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
        )

        print(f"⏱  {elapsed:.3f}s | {self.stats.report()}")
        return response.content

    def reset(self):
        self.stats = TokenStats()


# ── Usage ──────────────────────────────────────────────────────────────────────

from langchain_core.messages import HumanMessage

llm = TokenAwareLLM(model="gpt-4o-mini", token_budget=10_000)

for question in [
    "What is machine learning?",
    "Explain neural networks briefly.",
    "What is reinforcement learning?",
]:
    answer = llm.invoke([HumanMessage(content=question)])
    print(f"Q: {question}\nA: {answer[:80]}...\n")

print("\n📊 Final stats:", llm.stats.report())
```

---

### Method 4: LangGraph Node-Level Tracking

Track token usage **per node** inside a LangGraph `StateGraph` so you know exactly which step consumed the most tokens.

```python
# langgraph_token_tracking.py
from typing import TypedDict, Annotated, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END
import operator


# ── State ──────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages:         Annotated[List[BaseMessage], operator.add]
    token_usage:      dict   # accumulated across all nodes
    node_token_usage: dict   # per-node breakdown
    iteration:        int


# ── Helper ─────────────────────────────────────────────────────────────────────

llm = ChatOpenAI(model="gpt-4o-mini")

def _extract_usage(response) -> dict:
    u = response.response_metadata.get("token_usage", {})
    return {
        "prompt_tokens":     u.get("prompt_tokens", 0),
        "completion_tokens": u.get("completion_tokens", 0),
        "total_tokens":      u.get("total_tokens", 0),
    }

def _add_usage(current: dict, new: dict) -> dict:
    return {
        "prompt_tokens":     current.get("prompt_tokens", 0)     + new["prompt_tokens"],
        "completion_tokens": current.get("completion_tokens", 0) + new["completion_tokens"],
        "total_tokens":      current.get("total_tokens", 0)      + new["total_tokens"],
    }


# ── Nodes ──────────────────────────────────────────────────────────────────────

def reasoning_node(state: AgentState) -> AgentState:
    """Think step: reason about the user's request."""
    prompt = state["messages"] + [
        HumanMessage(content="Think step by step about how to answer the user's question.")
    ]
    response = llm.invoke(prompt)
    usage = _extract_usage(response)

    print(f"\n🧠 Reasoning node | tokens: {usage['total_tokens']}")
    return {
        **state,
        "messages":         state["messages"] + [response],
        "token_usage":      _add_usage(state["token_usage"], usage),
        "node_token_usage": {
            **state["node_token_usage"],
            f"reasoning_{state['iteration']}": usage,
        },
        "iteration": state["iteration"] + 1,
    }


def answer_node(state: AgentState) -> AgentState:
    """Answer step: formulate the final response."""
    prompt = state["messages"] + [
        HumanMessage(content="Now provide a clear, concise final answer.")
    ]
    response = llm.invoke(prompt)
    usage = _extract_usage(response)

    print(f"💬 Answer node    | tokens: {usage['total_tokens']}")
    return {
        **state,
        "messages":         state["messages"] + [response],
        "token_usage":      _add_usage(state["token_usage"], usage),
        "node_token_usage": {
            **state["node_token_usage"],
            f"answer_{state['iteration']}": usage,
        },
        "iteration": state["iteration"] + 1,
    }


def should_continue(state: AgentState) -> str:
    """Route: run reasoning → answer, then stop."""
    return "answer" if state["iteration"] == 1 else END


# ── Graph ──────────────────────────────────────────────────────────────────────

workflow = StateGraph(AgentState)
workflow.add_node("reasoning", reasoning_node)
workflow.add_node("answer",    answer_node)

workflow.set_entry_point("reasoning")
workflow.add_conditional_edges("reasoning", should_continue, {"answer": "answer", END: END})
workflow.add_edge("answer", END)

agent = workflow.compile()

# ── Run ────────────────────────────────────────────────────────────────────────

initial_state: AgentState = {
    "messages":         [HumanMessage(content="What are the main benefits of using LangGraph?")],
    "token_usage":      {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    "node_token_usage": {},
    "iteration":        0,
}

result = agent.invoke(initial_state)

print("\n" + "="*55)
print("📊 Token Usage Summary")
print("="*55)
print(f"Total prompt tokens:     {result['token_usage']['prompt_tokens']}")
print(f"Total completion tokens: {result['token_usage']['completion_tokens']}")
print(f"Grand total tokens:      {result['token_usage']['total_tokens']}")
print("\nPer-node breakdown:")
for node, usage in result["node_token_usage"].items():
    print(f"  {node:30s} → {usage['total_tokens']} tokens")
```

---

## LangSmith Integration

### Basic LangSmith Setup

LangSmith tracing is enabled with three environment variables — no code changes required.

```python
# langsmith_basic_setup.py
import os
from dotenv import load_dotenv
load_dotenv()

# These env vars activate LangSmith automatically
os.environ["LANGCHAIN_TRACING_V2"]  = "true"
os.environ["LANGCHAIN_ENDPOINT"]    = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"]     = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"]     = "my-langgraph-agent"   # shows in dashboard

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(model="gpt-4o-mini")
response = llm.invoke([HumanMessage(content="Hello, LangSmith!")])
print(response.content)
# ✅ Visit https://smith.langchain.com → your project to see the trace
```

---

### Custom Tracing with LangSmith

Use `@traceable` to wrap any function and send custom metadata, tags, and token counts to LangSmith.

```python
# langsmith_custom_tracing.py
import os
from dotenv import load_dotenv
load_dotenv()

from langsmith import traceable, Client
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from typing import List, Dict, Any

client = Client()
llm    = ChatOpenAI(model="gpt-4o-mini")


@traceable(
    name="reasoning-step",
    tags=["reasoning", "langgraph"],
    metadata={"step": "reasoning", "version": "1.0"},
)
def reasoning_step(query: str, context: str = "") -> Dict[str, Any]:
    """Decorated function — automatically traced in LangSmith."""
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nThink carefully before answering."
    response = llm.invoke([HumanMessage(content=prompt)])

    usage = response.response_metadata.get("token_usage", {})
    return {
        "reasoning":          response.content,
        "prompt_tokens":      usage.get("prompt_tokens", 0),
        "completion_tokens":  usage.get("completion_tokens", 0),
        "total_tokens":       usage.get("total_tokens", 0),
    }


@traceable(
    name="answer-generation",
    tags=["answer", "langgraph"],
    metadata={"step": "answer", "version": "1.0"},
)
def generate_answer(query: str, reasoning: str) -> Dict[str, Any]:
    prompt = f"Based on this reasoning:\n{reasoning}\n\nProvide a concise answer to: {query}"
    response = llm.invoke([HumanMessage(content=prompt)])

    usage = response.response_metadata.get("token_usage", {})
    return {
        "answer":             response.content,
        "prompt_tokens":      usage.get("prompt_tokens", 0),
        "completion_tokens":  usage.get("completion_tokens", 0),
        "total_tokens":       usage.get("total_tokens", 0),
    }


@traceable(
    name="full-agent-pipeline",
    tags=["agent", "production"],
    metadata={"pipeline_version": "2.0", "environment": "production"},
)
def run_agent_pipeline(query: str) -> Dict[str, Any]:
    """Top-level trace that nests the child traces above."""
    reasoning_result = reasoning_step(query, context="You are a helpful AI assistant.")
    answer_result    = generate_answer(query, reasoning_result["reasoning"])

    total_tokens = reasoning_result["total_tokens"] + answer_result["total_tokens"]
    return {
        "query":         query,
        "reasoning":     reasoning_result["reasoning"],
        "answer":        answer_result["answer"],
        "total_tokens":  total_tokens,
        "token_breakdown": {
            "reasoning": reasoning_result,
            "answer":    answer_result,
        },
    }


if __name__ == "__main__":
    result = run_agent_pipeline("What are the advantages of using LangSmith for AI monitoring?")
    print(f"\nAnswer: {result['answer']}")
    print(f"Total tokens used: {result['total_tokens']}")
    print("\n✅ Check https://smith.langchain.com for the full trace.")
```

---

### LangSmith Datasets & Evaluation

Create datasets and run evaluations programmatically to track quality over time.

```python
# langsmith_evaluation.py
import os
from dotenv import load_dotenv
load_dotenv()

from langsmith import Client
from langsmith.evaluation import evaluate, LangChainStringEvaluator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from typing import Dict

client = Client()
llm    = ChatOpenAI(model="gpt-4o-mini")


# ── 1. Create (or reuse) a dataset ────────────────────────────────────────────

DATASET_NAME = "agent-qa-dataset"

examples = [
    {"input": "What is LangGraph?",         "output": "LangGraph is a library for building stateful, multi-actor applications with LLMs."},
    {"input": "What is LangSmith?",         "output": "LangSmith is a platform for debugging, testing, and monitoring LLM applications."},
    {"input": "How do agents work in AI?",  "output": "AI agents use LLMs to reason, plan, and take actions to accomplish goals."},
]

if not client.has_dataset(dataset_name=DATASET_NAME):
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="QA pairs for agent evaluation",
    )
    client.create_examples(
        inputs =[{"question": ex["input"]}  for ex in examples],
        outputs=[{"answer":   ex["output"]} for ex in examples],
        dataset_id=dataset.id,
    )
    print(f"✅ Dataset '{DATASET_NAME}' created with {len(examples)} examples.")
else:
    print(f"ℹ️  Dataset '{DATASET_NAME}' already exists.")


# ── 2. Define the function to evaluate ────────────────────────────────────────

def agent_function(inputs: Dict) -> Dict:
    """The function under test. Must accept dict and return dict."""
    question = inputs["question"]
    response = llm.invoke([HumanMessage(content=question)])
    usage    = response.response_metadata.get("token_usage", {})
    return {
        "answer":       response.content,
        "total_tokens": usage.get("total_tokens", 0),
    }


# ── 3. Run evaluation ─────────────────────────────────────────────────────────

evaluators = [
    LangChainStringEvaluator("qa",        config={"llm": ChatOpenAI(model="gpt-4o-mini")}),
    LangChainStringEvaluator("criteria",  config={
        "criteria": {"conciseness": "Is the response concise and to the point?"},
        "llm": ChatOpenAI(model="gpt-4o-mini"),
    }),
]

results = evaluate(
    agent_function,
    data=DATASET_NAME,
    evaluators=evaluators,
    experiment_prefix="agent-v1",
    metadata={"model": "gpt-4o-mini", "version": "1.0"},
)

print(f"\n📊 Evaluation complete. Results: {results}")
print("✅ View detailed results at https://smith.langchain.com")
```

---

### LangSmith Dashboard Metrics

Query LangSmith programmatically to pull run-level metrics for reporting.

```python
# langsmith_metrics.py
import os
from dotenv import load_dotenv
load_dotenv()

from langsmith import Client
from datetime import datetime, timedelta
from typing import Dict, List
import statistics

client = Client()


def get_project_metrics(project_name: str, days: int = 7) -> Dict:
    """Fetch aggregated token and latency metrics for a LangSmith project."""
    start_time = datetime.utcnow() - timedelta(days=days)

    runs = list(client.list_runs(
        project_name=project_name,
        start_time=start_time,
        run_type="llm",          # only LLM calls, not chains/tools
        limit=500,
    ))

    if not runs:
        print(f"No runs found in project '{project_name}' for the last {days} days.")
        return {}

    total_tokens_list:      List[int]   = []
    prompt_tokens_list:     List[int]   = []
    completion_tokens_list: List[int]   = []
    latency_list:           List[float] = []
    errors = 0

    for run in runs:
        # Token usage
        usage = (run.prompt_tokens or 0), (run.completion_tokens or 0)
        prompt_tokens_list.append(usage[0])
        completion_tokens_list.append(usage[1])
        total_tokens_list.append(usage[0] + usage[1])

        # Latency
        if run.end_time and run.start_time:
            latency_list.append((run.end_time - run.start_time).total_seconds())

        # Errors
        if run.error:
            errors += 1

    return {
        "project":            project_name,
        "period_days":        days,
        "total_runs":         len(runs),
        "error_count":        errors,
        "success_rate":       f"{((len(runs) - errors) / len(runs) * 100):.1f}%",
        "token_stats": {
            "total_prompt_tokens":      sum(prompt_tokens_list),
            "total_completion_tokens":  sum(completion_tokens_list),
            "grand_total_tokens":       sum(total_tokens_list),
            "avg_tokens_per_run":       round(statistics.mean(total_tokens_list), 1),
            "max_tokens_single_run":    max(total_tokens_list),
        },
        "latency_stats": {
            "avg_seconds":    round(statistics.mean(latency_list), 3) if latency_list else 0,
            "median_seconds": round(statistics.median(latency_list), 3) if latency_list else 0,
            "max_seconds":    round(max(latency_list), 3) if latency_list else 0,
        },
    }


def print_metrics_report(metrics: Dict):
    """Pretty-print the metrics dict."""
    if not metrics:
        return
    print("\n" + "="*55)
    print(f"📊 LangSmith Metrics — {metrics['project']}")
    print("="*55)
    print(f"Period:        Last {metrics['period_days']} days")
    print(f"Total runs:    {metrics['total_runs']}")
    print(f"Errors:        {metrics['error_count']}")
    print(f"Success rate:  {metrics['success_rate']}")

    t = metrics["token_stats"]
    print(f"\nToken Usage:")
    print(f"  Prompt tokens:      {t['total_prompt_tokens']:,}")
    print(f"  Completion tokens:  {t['total_completion_tokens']:,}")
    print(f"  Grand total:        {t['grand_total_tokens']:,}")
    print(f"  Avg per run:        {t['avg_tokens_per_run']:,}")
    print(f"  Max single run:     {t['max_tokens_single_run']:,}")

    l = metrics["latency_stats"]
    print(f"\nLatency:")
    print(f"  Average:  {l['avg_seconds']}s")
    print(f"  Median:   {l['median_seconds']}s")
    print(f"  Max:      {l['max_seconds']}s")


if __name__ == "__main__":
    metrics = get_project_metrics("my-langgraph-agent", days=7)
    print_metrics_report(metrics)
```

---

## Full Agent Example with Everything Combined

A production-ready LangGraph agent that combines:
- Node-level token tracking
- LangSmith `@traceable` decorators
- Token budget enforcement
- Structured state

```python
# full_agent.py
import os
import operator
from typing import TypedDict, Annotated, List, Optional
from dotenv import load_dotenv
load_dotenv()

# Activate LangSmith
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_PROJECT",    "full-agent-demo")

from langsmith import traceable
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END
from token_callback import TokenUsageCallbackHandler   # Method 1 above


# ── Configuration ──────────────────────────────────────────────────────────────

TOKEN_BUDGET = 5_000   # hard limit per agent run

tracker = TokenUsageCallbackHandler()
llm     = ChatOpenAI(model="gpt-4o-mini", callbacks=[tracker])


# ── State ──────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages:         Annotated[List[BaseMessage], operator.add]
    token_budget:     int
    token_usage:      dict
    node_token_usage: dict
    current_node:     str
    iteration:        int
    final_answer:     Optional[str]


# ── Nodes ──────────────────────────────────────────────────────────────────────

@traceable(name="planning-node", tags=["planning"])
def planning_node(state: AgentState) -> AgentState:
    """Plan how to answer the user's question."""
    if state["token_usage"].get("total_tokens", 0) >= state["token_budget"]:
        raise RuntimeError("Token budget exceeded in planning node.")

    response = llm.invoke(
        state["messages"] + [HumanMessage(content="Create a concise plan to answer the user's question.")]
    )

    usage = response.response_metadata.get("token_usage", {})
    current_total = {
        "prompt_tokens":     state["token_usage"].get("prompt_tokens", 0)     + usage.get("prompt_tokens", 0),
        "completion_tokens": state["token_usage"].get("completion_tokens", 0) + usage.get("completion_tokens", 0),
        "total_tokens":      state["token_usage"].get("total_tokens", 0)      + usage.get("total_tokens", 0),
    }

    print(f"\n📋 Planning | tokens this call: {usage.get('total_tokens', 0)} | running total: {current_total['total_tokens']}")

    return {
        **state,
        "messages":         state["messages"] + [response],
        "token_usage":      current_total,
        "node_token_usage": {**state["node_token_usage"], "planning": usage},
        "current_node":     "planning",
        "iteration":        state["iteration"] + 1,
    }


@traceable(name="execution-node", tags=["execution"])
def execution_node(state: AgentState) -> AgentState:
    """Execute the plan and generate a detailed answer."""
    if state["token_usage"].get("total_tokens", 0) >= state["token_budget"]:
        raise RuntimeError("Token budget exceeded in execution node.")

    response = llm.invoke(
        state["messages"] + [HumanMessage(content="Execute the plan and provide a comprehensive answer.")]
    )

    usage = response.response_metadata.get("token_usage", {})
    current_total = {
        "prompt_tokens":     state["token_usage"].get("prompt_tokens", 0)     + usage.get("prompt_tokens", 0),
        "completion_tokens": state["token_usage"].get("completion_tokens", 0) + usage.get("completion_tokens", 0),
        "total_tokens":      state["token_usage"].get("total_tokens", 0)      + usage.get("total_tokens", 0),
    }

    print(f"⚙️  Execution | tokens this call: {usage.get('total_tokens', 0)} | running total: {current_total['total_tokens']}")

    return {
        **state,
        "messages":         state["messages"] + [response],
        "token_usage":      current_total,
        "node_token_usage": {**state["node_token_usage"], "execution": usage},
        "current_node":     "execution",
        "iteration":        state["iteration"] + 1,
        "final_answer":     response.content,
    }


@traceable(name="review-node", tags=["review"])
def review_node(state: AgentState) -> AgentState:
    """Review and refine the answer for quality."""
    if state["token_usage"].get("total_tokens", 0) >= state["token_budget"]:
        print("⚠️  Token budget reached — skipping review.")
        return {**state, "current_node": "review"}

    response = llm.invoke(
        state["messages"] + [HumanMessage(content="Review the answer above. If it's good, say 'APPROVED'. Otherwise, provide a brief improvement.")]
    )

    usage = response.response_metadata.get("token_usage", {})
    current_total = {
        "prompt_tokens":     state["token_usage"].get("prompt_tokens", 0)     + usage.get("prompt_tokens", 0),
        "completion_tokens": state["token_usage"].get("completion_tokens", 0) + usage.get("completion_tokens", 0),
        "total_tokens":      state["token_usage"].get("total_tokens", 0)      + usage.get("total_tokens", 0),
    }

    print(f"🔍 Review    | tokens this call: {usage.get('total_tokens', 0)} | running total: {current_total['total_tokens']}")

    return {
        **state,
        "messages":         state["messages"] + [response],
        "token_usage":      current_total,
        "node_token_usage": {**state["node_token_usage"], "review": usage},
        "current_node":     "review",
        "iteration":        state["iteration"] + 1,
    }


# ── Routing ────────────────────────────────────────────────────────────────────

def route_after_review(state: AgentState) -> str:
    last_content = state["messages"][-1].content if state["messages"] else ""
    if "APPROVED" in last_content.upper() or state["iteration"] >= 4:
        return END
    return "execution"   # loop back for one more refinement


# ── Graph ──────────────────────────────────────────────────────────────────────

workflow = StateGraph(AgentState)
workflow.add_node("planning",  planning_node)
workflow.add_node("execution", execution_node)
workflow.add_node("review",    review_node)

workflow.set_entry_point("planning")
workflow.add_edge("planning",  "execution")
workflow.add_edge("execution", "review")
workflow.add_conditional_edges("review", route_after_review, {"execution": "execution", END: END})

agent = workflow.compile()


# ── Run ────────────────────────────────────────────────────────────────────────

@traceable(name="agent-run", tags=["production"], metadata={"version": "1.0"})
def run_agent(query: str) -> dict:
    initial_state: AgentState = {
        "messages":         [HumanMessage(content=query)],
        "token_budget":     TOKEN_BUDGET,
        "token_usage":      {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "node_token_usage": {},
        "current_node":     "start",
        "iteration":        0,
        "final_answer":     None,
    }
    return agent.invoke(initial_state)


if __name__ == "__main__":
    result = run_agent("Explain the key benefits of using LangGraph for building AI agents.")

    print("\n" + "="*55)
    print("✅ Final Answer")
    print("="*55)
    print(result["final_answer"])

    print("\n" + "="*55)
    print("📊 Token Usage by Node")
    print("="*55)
    for node, usage in result["node_token_usage"].items():
        print(f"  {node:12s}: {usage.get('total_tokens', 0):>5} tokens")
    print(f"\n  {'TOTAL':12s}: {result['token_usage']['total_tokens']:>5} tokens")
    print(f"  Budget used: {result['token_usage']['total_tokens']}/{TOKEN_BUDGET} "
          f"({result['token_usage']['total_tokens']/TOKEN_BUDGET*100:.1f}%)")
    print("\n🔗 View traces at https://smith.langchain.com")
```

---

## Best Practices

### 1. Always set a token budget
```python
TOKEN_BUDGET = 10_000
if state["token_usage"]["total_tokens"] >= TOKEN_BUDGET:
    raise RuntimeError("Token budget exceeded")
```

### 2. Track tokens at every node in LangGraph
Store both a running total (`token_usage`) and a per-node breakdown (`node_token_usage`) in your state so you can pinpoint expensive steps.

### 3. Use meaningful LangSmith project names
```python
os.environ["LANGCHAIN_PROJECT"] = f"agent-{environment}-{version}"
# e.g. "agent-production-v2"
```

### 4. Add tags and metadata to traces
```python
@traceable(
    name="my-node",
    tags=["production", "v2"],
    metadata={"model": "gpt-4o-mini", "feature_flag": "new_reasoning"},
)
```

### 5. Monitor costs, not just counts

| Model       | Input (per 1K tokens) | Output (per 1K tokens) |
|-------------|----------------------|------------------------|
| gpt-4o      | $0.0050              | $0.0150                |
| gpt-4o-mini | $0.00015             | $0.00060               |
| gpt-4-turbo | $0.0100              | $0.0300                |

```python
estimated_cost = (prompt_tokens / 1000 * 0.00015) + (completion_tokens / 1000 * 0.00060)
```

### 6. Use LangSmith for A/B testing
Run experiments with different prompts or models using `experiment_prefix` in `evaluate()`, then compare results side-by-side in the dashboard.

### 7. Set up alerts
In the LangSmith dashboard → **Rules** → create alerts for:
- Error rate > 5%
- P95 latency > 10s
- Average tokens per run > your threshold

---

## Quick Reference

```bash
# Enable LangSmith tracing (shell)
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=your_key
export LANGCHAIN_PROJECT=my-project

# Install all dependencies
pip install langgraph langchain langchain-openai langsmith python-dotenv
```

```python
# Minimal token monitoring snippet
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm      = ChatOpenAI(model="gpt-4o-mini")
response = llm.invoke([HumanMessage(content="Hello!")])
usage    = response.response_metadata["token_usage"]
print(f"Tokens used: {usage['total_tokens']}")
```

---

*Happy building! 🚀 For questions, open an issue or check the [LangChain docs](https://python.langchain.com) and [LangSmith docs](https://docs.smith.langchain.com).*
