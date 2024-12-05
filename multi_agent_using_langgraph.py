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
