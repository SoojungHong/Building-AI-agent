from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from typing import Dict, Callable

# Initialize the OpenAI LLM
openai_api_key = "your-openai-api-key"
llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

# Initialize a vector store
def create_vector_store() -> FAISS:
    # Example documents for vector store
    documents = [
        "The capital of France is Paris.",
        "Python is a popular programming language.",
        "The Turing Test was proposed by Alan Turing.",
    ]

    # Use OpenAI embeddings to convert text to vectors
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = FAISS.from_texts(documents, embeddings)
    return vector_store

vector_store = create_vector_store()

# Define a generic function that handles LLM + vector store calls
def execute_llm_with_vector_store(prompt: str) -> str:
    # Set up a retrieval-based QA chain
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    # Execute the LLM chain with the prompt
    response = qa_chain.run(prompt)
    return response

# Define a graph structure with nodes
class Graph:
    def __init__(self):
        self.nodes: Dict[str, Callable[[str], str]] = {}

    def add_node(self, name: str, method: Callable[[str], str]):
        """Add a node with a method to the graph."""
        self.nodes[name] = method

    def call_node(self, name: str, prompt: str) -> str:
        """Call a specific node with a prompt."""
        if name not in self.nodes:
            raise ValueError(f"Node '{name}' does not exist in the graph.")
        return self.nodes[name](prompt)

# Initialize the graph and nodes
graph = Graph()

# Add a node to the graph that calls the LLM with a vector store
graph.add_node("node1", execute_llm_with_vector_store)

def custom_node(prompt: str) -> str:
    return f"Custom Node received: {prompt}"

graph.add_node("custom_node", custom_node)

response = graph.call_node("custom_node", "Hello, custom node!")
print(response)


# Example usage
if __name__ == "__main__":
    # Call the node with a prompt
    prompt = "What is the capital of France?"
    response = graph.call_node("node1", prompt)
    print(f"Response from node1: {response}")

    # Call the node with another prompt
    prompt = "Who proposed the Turing Test?"
    response = graph.call_node("node1", prompt)
    print(f"Response from node1: {response}")
