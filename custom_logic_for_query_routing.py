"""
To implement Custom Logic for Query Routing in the Coordinator_Agent, you can define a routing mechanism that analyzes the user query and determines which agent(s) should handle it. 
For instance, routing logic can use keyword matching, natural language processing (NLP), or embeddings similarity.

Here’s an example where the Coordinator_Agent routes the query to the appropriate agent(s) based on keywords in the query:
"""

from typing import List, Dict
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langgraph.agents import Agent, LangGraph

# Initialize Vector Stores (similar to before)
vector_store_1 = FAISS.load_local("path_to_faiss_store_1", OpenAIEmbeddings())
vector_store_2 = FAISS.load_local("path_to_faiss_store_2", OpenAIEmbeddings())
vector_store_3 = FAISS.load_local("path_to_faiss_store_3", OpenAIEmbeddings())

# Define Agents
llm = ChatOpenAI(model="gpt-4")  # Replace with your LLM

agent_1 = Agent(
    name="Agent_1",
    chain=RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store_1.as_retriever(),
        chain_type="stuff",
        return_source_documents=True,
    ),
    description="Handles queries about Document Set 1: Science topics.",
)

agent_2 = Agent(
    name="Agent_2",
    chain=RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store_2.as_retriever(),
        chain_type="stuff",
        return_source_documents=True,
    ),
    description="Handles queries about Document Set 2: Business topics.",
)

agent_3 = Agent(
    name="Agent_3",
    chain=RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store_3.as_retriever(),
        chain_type="stuff",
        return_source_documents=True,
    ),
    description="Handles queries about Document Set 3: Technology topics.",
)

# Custom Routing Logic
class CustomCoordinator:
    def __init__(self, agents: List[Agent]):
        self.agents = {agent.name: agent}
        self.keywords = {
            "science": "Agent_1",
            "biology": "Agent_1",
            "business": "Agent_2",
            "finance": "Agent_2",
            "technology": "Agent_3",
            "software": "Agent_3",
        }

    def route_query(self, query: str) -> List[Agent]:
        """Determine which agent(s) should handle the query based on keywords."""
        selected_agents = []

        for keyword, agent_name in self.keywords.items():
            if keyword.lower() in query.lower():
                selected_agents.append(self.agents[agent_name])

        # If no specific match, send the query to all agents
        if not selected_agents:
            selected_agents = list(self.agents.values())

        return selected_agents

    def process_query(self, query: str) -> Dict[str, str]:
        """Process the query and gather responses from relevant agents."""
        relevant_agents = self.route_query(query)
        responses = {}

        for agent in relevant_agents:
            print(f"Routing query to {agent.name}...")
            response = agent.run(query)  # Call the agent's chain
            responses[agent.name] = response

        return responses

# Instantiate the Coordinator
coordinator = CustomCoordinator([agent_1, agent_2, agent_3])

# Interact with the system
user_query = "What are the latest advancements in software engineering?"
responses = coordinator.process_query(user_query)

# Display responses
print("Responses from Agents:")
for agent_name, response in responses.items():
    print(f"\n{agent_name}: {response}")


"""
Share


You said:
I have several vector stores with heterogenous documents. To use each vector stores, I want to build a multi agents. I will use LangGraph to build a multi agent. Show me the python code.
ChatGPT said:
ChatGPT
To build a multi-agent system using LangGraph where each agent works with a different vector store containing heterogeneous documents, you can leverage the modular nature of LangChain and LangGraph. Below is an example Python script to achieve this.

Steps to Create Multi-Agent with LangGraph
Initialize Vector Stores: Load your vector stores (e.g., FAISS, Pinecone, or any other) with heterogeneous documents.
Define Agents: Each agent interacts with one specific vector store and is configured with the appropriate retriever and LLM.
Build LangGraph: Use LangGraph to manage the communication between agents.
Handle Agent Coordination: Agents share information when needed, or a "controller" agent can coordinate tasks.
Here’s a sample implementation:

Python Code
python
Copy code
from langchain.vectorstores import FAISS, Pinecone
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langgraph.agents import Agent, LangGraph
from langchain.embeddings import OpenAIEmbeddings

# Initialize Vector Stores
vector_store_1 = FAISS.load_local("path_to_faiss_store_1", OpenAIEmbeddings())
vector_store_2 = Pinecone.from_existing_index("pinecone_index_2", OpenAIEmbeddings())
vector_store_3 = FAISS.load_local("path_to_faiss_store_3", OpenAIEmbeddings())

# Define Agents
llm = ChatOpenAI(model="gpt-4")  # Replace with the LLM of your choice

agent_1 = Agent(
    name="Agent_1",
    chain=RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store_1.as_retriever(),
        chain_type="stuff",
        return_source_documents=True,
    ),
    description="Handles information related to Document Set 1.",
)

agent_2 = Agent(
    name="Agent_2",
    chain=RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store_2.as_retriever(),
        chain_type="stuff",
        return_source_documents=True,
    ),
    description="Handles information related to Document Set 2.",
)

agent_3 = Agent(
    name="Agent_3",
    chain=RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store_3.as_retriever(),
        chain_type="stuff",
        return_source_documents=True,
    ),
    description="Handles information related to Document Set 3.",
)

# Create LangGraph
multi_agent_system = LangGraph(
    agents=[agent_1, agent_2, agent_3],
    central_agent=Agent(
        name="Coordinator_Agent",
        llm=llm,
        description="Coordinates tasks and determines which agent to query based on user input.",
    ),
)

# Interact with the Multi-Agent System
user_query = "Find information about topic X and summarize across all document sets."
response = multi_agent_system.run(user_query)

print("Response from Multi-Agent System:")
print(response)
Key Points in the Code
Vector Stores: You can replace FAISS and Pinecone with any vector store supported by LangChain.
Agents: Each agent is configured with its own retriever and LLM to specialize in a specific domain.
Central Agent: The Coordinator_Agent determines which agent(s) to query based on the input. You can add custom logic here if needed.
LangGraph: This acts as the communication hub for the agents.
Dependencies
Install the required packages:
bash
Copy code
pip install langchain langgraph faiss-cpu pinecone-client openai
Expandability
Custom Logic for Query Routing: Enhance the central agent's logic to better route tasks based on keywords or input type.
Asynchronous Execution: Use asynchronous calls for better performance when querying multiple agents simultaneously.
Let me know if you need further customization or a deeper dive into any part of the code!






You said:
Show me the code that can make Custom Logic for Query Routing.
ChatGPT said:
ChatGPT
To implement Custom Logic for Query Routing in the Coordinator_Agent, you can define a routing mechanism that analyzes the user query and determines which agent(s) should handle it. For instance, routing logic can use keyword matching, natural language processing (NLP), or embeddings similarity.

Here’s an example where the Coordinator_Agent routes the query to the appropriate agent(s) based on keywords in the query:

Python Code for Custom Query Routing
python
Copy code
from typing import List, Dict
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langgraph.agents import Agent, LangGraph

# Initialize Vector Stores (similar to before)
vector_store_1 = FAISS.load_local("path_to_faiss_store_1", OpenAIEmbeddings())
vector_store_2 = FAISS.load_local("path_to_faiss_store_2", OpenAIEmbeddings())
vector_store_3 = FAISS.load_local("path_to_faiss_store_3", OpenAIEmbeddings())

# Define Agents
llm = ChatOpenAI(model="gpt-4")  # Replace with your LLM

agent_1 = Agent(
    name="Agent_1",
    chain=RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store_1.as_retriever(),
        chain_type="stuff",
        return_source_documents=True,
    ),
    description="Handles queries about Document Set 1: Science topics.",
)

agent_2 = Agent(
    name="Agent_2",
    chain=RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store_2.as_retriever(),
        chain_type="stuff",
        return_source_documents=True,
    ),
    description="Handles queries about Document Set 2: Business topics.",
)

agent_3 = Agent(
    name="Agent_3",
    chain=RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store_3.as_retriever(),
        chain_type="stuff",
        return_source_documents=True,
    ),
    description="Handles queries about Document Set 3: Technology topics.",
)

# Custom Routing Logic
class CustomCoordinator:
    def __init__(self, agents: List[Agent]):
        self.agents = {agent.name: agent}
        self.keywords = {
            "science": "Agent_1",
            "biology": "Agent_1",
            "business": "Agent_2",
            "finance": "Agent_2",
            "technology": "Agent_3",
            "software": "Agent_3",
        }

    def route_query(self, query: str) -> List[Agent]:
        """Determine which agent(s) should handle the query based on keywords."""
        selected_agents = []

        for keyword, agent_name in self.keywords.items():
            if keyword.lower() in query.lower():
                selected_agents.append(self.agents[agent_name])

        # If no specific match, send the query to all agents
        if not selected_agents:
            selected_agents = list(self.agents.values())

        return selected_agents

    def process_query(self, query: str) -> Dict[str, str]:
        """Process the query and gather responses from relevant agents."""
        relevant_agents = self.route_query(query)
        responses = {}

        for agent in relevant_agents:
            print(f"Routing query to {agent.name}...")
            response = agent.run(query)  # Call the agent's chain
            responses[agent.name] = response

        return responses

# Instantiate the Coordinator
coordinator = CustomCoordinator([agent_1, agent_2, agent_3])

# Interact with the system
user_query = "What are the latest advancements in software engineering?"
responses = coordinator.process_query(user_query)

# Display responses
print("Responses from Agents:")
for agent_name, response in responses.items():
    print(f"\n{agent_name}: {response}")
Explanation of the Custom Coordinator
Keyword-Based Routing:

The route_query method checks for specific keywords in the query and routes it to the corresponding agent(s).
For example, a query containing "finance" will be routed to Agent_2 because it is associated with business topics.
Fallback to All Agents:

If no keywords match, the query is sent to all agents by default.
Processing the Query:

The process_query method collects responses from the agents identified by the route_query method.
Each agent runs its own RetrievalQA chain to answer the query.
Flexible Configuration:

You can expand the self.keywords dictionary with additional keywords or even use embeddings to match queries to agents based on semantic similarity.
"""
