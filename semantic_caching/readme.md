How to make Agent response time faster? 

Getting from 2+ minutes to under 1 minute is usually realistic. In multi-step LangGraph agents, the time almost always comes from the number of sequential LLM calls multiplied by the tokens generated per call. Tools are rarely the main cost. Work through these in order.

1. Measure where the time goes

Turn on LangSmith tracing (LANGSMITH_TRACING=true) and look at one slow run. Note three things:

How many LLM calls happen in sequence? Ten calls at 10s each is your 2 minutes.
Which calls generate the most output tokens? Output tokens are far slower than input tokens.
Are any tools slow, or is the agent retrying because a tool failed or returned confusing output?

The fixes below map directly to what you find.

2. Cut the number of sequential LLM calls (biggest win)
Replace LLM routing with code. If a node only decides “go to A or B”, use a conditional edge with plain Python logic or a cheap classifier, not a full reasoning call.
Plan once, then execute. A ReAct loop calls the LLM after every tool result. A plan-and-execute pattern makes one planning call, runs the tools, and makes one synthesis call.
Merge steps. Two consecutive LLM nodes, such as “extract” then “summarize”, can often be one call with structured output.
Stop loops. Set a recursion_limit, write clearer tool descriptions, and return clean tool errors so the agent doesn’t retry 3 or 4 times.
3. Run independent work in parallel

If steps don’t depend on each other, fan them out:

python
# Parallel branches in LangGraph: both nodes run in the same superstep
builder.add_edge("planner", "search_docs")
builder.add_edge("planner", "query_db")
builder.add_edge(["search_docs", "query_db"], "synthesize")
Use the Send API for dynamic fan-out, such as one subtask per item.
Enable parallel tool calling, which most models support by default with bind_tools, and make tools async so they actually run concurrently.
Use ainvoke and asyncio.gather rather than sync calls inside nodes.
4. Use the right model for each step
Use a small, fast model (Haiku-class or mini-class) for routing, extraction, classification, and summarizing tool output.
Reserve the big model for the one or two steps that need real reasoning.
If you use a reasoning or thinking model, lower the reasoning effort or thinking budget on intermediate steps. This alone can halve latency.
5. Shrink tokens
Output: set max_tokens, ask for concise answers, and use structured output for intermediate steps instead of prose.
Input: trim or summarize message history between steps, truncate large tool results before they go back to the LLM, and enable prompt caching for long, stable system prompts.
6. Speed up the tools

Add caching (for repeated retrievals and API lookups), connection pooling, timeouts, and smaller k in vector search. Check that your DB or vector store and the LLM provider are in a nearby region.

7. Stream to fix perceived latency

Even once the real time is under a minute, stream progress so users see activity immediately:

python
async for chunk in graph.astream(inputs, stream_mode=["updates", "messages"]):
    ...
Typical result

Take an agent that does 8 sequential calls of about 15s each. After moving routing to code, running 3 lookups in parallel, putting intermediate steps on a small model, and capping output, it often ends up at 3–4 calls of 5–10s each, which is roughly 30–45s.

If you share your graph structure (nodes and edges) or a trace breakdown, I can point to the specific changes that will save the most time.
