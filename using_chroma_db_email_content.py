Step 1: Install ChromaDB
If you haven't installed ChromaDB, you can do so using:

pip install chromadb
Step 2: Prepare Your Email Data with Metadata
When preparing your email data, include metadata fields such as date, sender, subject, category, etc. Here's an example:

emails = [
    {
        "content": "Details of the M&A deal between Company A and Company B...",
        "metadata": {
            "subject": "M&A Deal",
            "date": "2024-11-13",
            "sender": "john.doe@example.com",
            "category": "Finance",
            "company": "Company A"
        }
    },
    {
        "content": "Update on the merger process for Company B...",
        "metadata": {
            "subject": "Merger Update",
            "date": "2024-11-10",
            "sender": "jane.smith@example.com",
            "category": "Corporate",
            "company": "Company B"
        }
    },
    # Add more email data...
]
Step 3: Initialize ChromaDB and Create a Collection
You need to set up ChromaDB and create a collection where your documents (emails) will be stored.

import chromadb

# Initialize ChromaDB client
client = chromadb.Client()

# Create or get a collection for storing your email data
collection = client.create_collection(name="emails")
Step 4: Add Email Documents with Metadata to ChromaDB
You will insert your email content into the ChromaDB collection, associating each email with its respective metadata.

# Loop through your emails and add them to the collection
for idx, email in enumerate(emails):
    collection.add(
        ids=[f"email_{idx}"],  # Unique ID for each email
        documents=[email["content"]],  # The content of the email
        metadatas=[email["metadata"]]  # Metadata associated with the email
    )
Step 5: Perform a Search with Metadata Filtering
You can now query the ChromaDB collection and use metadata filters to narrow down your search results.

# Define your query and metadata filter
query = "What are the latest M&A deals?"
metadata_filter = {
    "date": {"$gte": "2024-01-01"},  # Example: Filter for dates greater than or equal to 2024-01-01
    "category": "Finance"             # Example: Filter by category "Finance"
}

# Perform a search in ChromaDB
results = collection.query(
    query_texts=[query],  # The query to search for
    n_results=10,         # Number of results to return
    where=metadata_filter # Metadata filtering criteria
)

# Process and display the results
for result in results["documents"]:
    print(result["content"], result["metadata"])
Explanation of Metadata Filtering
where Clause: In ChromaDB, you can use the where parameter to specify filtering criteria based on your metadata. The filter uses a dictionary format, where keys are metadata fields and values are the conditions to be met.
Supported Conditions: You can use operators like $gte (greater than or equal), $lte (less than or equal), $eq (equal), etc., for filtering. This gives you flexibility in narrowing down results.
Example: More Complex Metadata Filtering
If you want to combine multiple filters, you can do so like this:

# More complex filter combining multiple conditions
metadata_filter = {
    "date": {"$gte": "2024-01-01"},  # Filter by date
    "company": "Company A",           # Filter by company
    "sender": {"$eq": "john.doe@example.com"}  # Filter by sender
}

# Perform a query with the complex filter
results = collection.query(
    query_texts=["Details about M&A"],
    n_results=5,
    where=metadata_filter
)

# Display the filtered results
for result in results["documents"]:
    print(result["content"], result["metadata"])
Step 6: Use ChromaDB with LangChain
To make this setup part of a LangChain RAG system, you can configure LangChain to use ChromaDB as the vector store.

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# Initialize OpenAI Embeddings (or another embedding model)
embeddings = OpenAIEmbeddings()

# Connect LangChain to your ChromaDB collection
vector_store = Chroma(collection=collection, embedding_function=embeddings)

# Use LangChain's RetrievalQA chain
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

llm = OpenAI(temperature=0.7)  # Initialize your language model
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever()
)

# Answer questions with LangChain
query = "What are the recent updates on Company A?"
answer = qa_chain.run(query)
print(answer)
Summary
Metadata Storage: When adding documents to ChromaDB, store metadata alongside your content.
Metadata Filtering: Use the where parameter to apply filters based on metadata fields.
LangChain Integration: Connect ChromaDB to LangChain for seamless RAG functionality.
This approach allows you to efficiently search and filter documents using both semantic content and metadata, enhancing the relevance of the information retrieved.
