"""
use MemorySaver without defining a custom state by leveraging the system message to dynamically inject context about the current client. 

idea : per user (e.g., CA or RM's email), store the the latest entities -  client identifier, opportunity name 
example : "myname@abc.com" :
                    {
                     "client" : abc, 
                     "opportunity": 234
                     } 

workflow : 
1. per user 
2. detect the entities - client identifier, opportunity id, etc 
3. when prompt has 'this', 'that', etc
4. search the entity database
"""

from langgraph.graph import MessagesState, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, RemoveMessage
import re

# Sample client database
CLIENT_DATABASE = {
    "ABC": {
        "name": "Client ABC",
        "phone": "+1-555-0123",
        "email": "abc@example.com",
        "address": "123 Main St, New York"
    },
    "XYZ": {
        "name": "Client XYZ",
        "phone": "+1-555-0456",
        "email": "xyz@example.com",
        "address": "456 Oak Ave, Boston"
    }
}

# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)

def extract_client_from_messages(messages):
    """Extract the most recently mentioned client from conversation history"""
    # Search messages in reverse order (most recent first)
    for message in reversed(messages):
        content = message.content
        for client_key in CLIENT_DATABASE.keys():
            if client_key.upper() in content.upper():
                return client_key
    return None

# ToDo : extract_entity_from_messages(messages) , run it parallel? 
# instead of extract only client, extract entities given current user email 

def chatbot_node(state: MessagesState):
    """Chatbot node that tracks entities through message history"""
    messages = state["messages"]
    
    # Extract current client from conversation history
    current_client = extract_client_from_messages(messages)
    # ToDo : current_entity = extract_entity_from_messages(messages) 
    
    # Create dynamic system message with context
    system_content = f"""You are a helpful assistant with access to client information.

ENTITY TRACKING:
- Current client being discussed: {current_client if current_client else "None"}
- When users say "this client", "that client", "the client", "their phone", "they", etc., 
  they are referring to: **{current_client}**

Available clients: {', '.join(CLIENT_DATABASE.keys())}

Database:
{CLIENT_DATABASE}

INSTRUCTIONS:
1. If a user asks about "this client" or uses pronouns, interpret it as referring to {current_client}
2. Provide accurate information from the database
3. If no client is in context and user uses references like "this client", ask for clarification
"""
    
    # Build messages list with system message
    # Remove any old system messages to avoid duplication
    non_system_messages = [msg for msg in messages if not isinstance(msg, SystemMessage)]
    
    llm_messages = [SystemMessage(content=system_content)] + non_system_messages
    
    # Get response from LLM
    response = llm.invoke(llm_messages)
    
    return {"messages": [response]}

# Create the graph using built-in MessagesState
workflow = StateGraph(MessagesState)
workflow.add_node("chatbot", chatbot_node)
workflow.set_entry_point("chatbot")
workflow.set_finish_point("chatbot")

# Compile with MemorySaver
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

def chat(user_input, thread_id="default"):
    """Chat function with memory"""
    config = {"configurable": {"thread_id": thread_id}}
    
    result = app.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=config
    )
    
    return result["messages"][-1].content

# Example usage
if __name__ == "__main__":
    thread_id = "user_123"
    
    print("Chatbot: Hello! I can help you with client information.\n")
    
    # First query
    response1 = chat("Tell me Client ABC's information", thread_id)
    print(f"User: Tell me Client ABC's information")
    print(f"Chatbot: {response1}\n")
    
    # Second query - uses "this client"
    response2 = chat("What is the phone number of this client?", thread_id)
    print(f"User: What is the phone number of this client?")
    print(f"Chatbot: {response2}\n")
    
    # Third query - uses "their"
    response3 = chat("What's their email?", thread_id)
    print(f"User: What's their email?")
    print(f"Chatbot: {response3}\n")
    
    # Fourth query - switch to different client
    response4 = chat("Now tell me about Client XYZ", thread_id)
    print(f"User: Now tell me about Client XYZ")
    print(f"Chatbot: {response4}\n")
    
    # Fifth query - reference new client
    response5 = chat("What's this client's address?", thread_id)
    print(f"User: What's this client's address?")
    print(f"Chatbot: {response5}\n")
