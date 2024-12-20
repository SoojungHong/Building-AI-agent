"""
Steps to Improve Response Time
1. Batch Queries for Vector Search
Instead of searching for every query individually, you can precompute embeddings or batch the embedding generation to reduce latency.

2. Reduce Vector Store Retrieval Size
Set the k parameter (number of top results to retrieve) to a reasonable value (e.g., k=3) rather than fetching a large number of results.

3. Optimize Vector Store Backend
For vector stores, ensure:

Indexes are optimized (e.g., use approximate nearest neighbor methods like HNSW).
The database is on a high-performance infrastructure.
4. Use Faster Embedding Models
Switch to a faster embedding model (e.g., text-embedding-ada-002) if possible.

5. Cache Embeddings
Precompute embeddings for static queries/documents and store them. This avoids redundant embedding computation.

6. Optimize LLM Calls
Reduce the token size of input by trimming retrieved documents or using a summary chain.
Use a smaller model (e.g., gpt-3.5-turbo) if high accuracy isn’t required.

7. Enable Asynchronous Execution
Run vector search and LLM inference in parallel or asynchronously.

"""

"""
Optimized Code Example
"""
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import asyncio

# 1. Use a Faster Embedding Model
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")  # Faster and cost-effective

# 2. Initialize Vector Store
vector_store = FAISS.load_local("path_to_faiss_store", embeddings)

# 3. Optimize Retriever with a Lower 'k' Value
retriever = vector_store.as_retriever(search_kwargs={"k": 3})  # Fetch fewer top results

# 4. Define a Custom Prompt to Reduce Input Token Size
prompt_template = """
You are a helpful assistant. Answer the user's question concisely based on the provided documents.

Documents: {context}

Question: {question}
Answer:
"""
prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

# 5. Initialize Chat Model (Smaller Model for Faster Responses)
llm = ChatOpenAI(model="gpt-3.5-turbo")  # Use smaller model if speed > accuracy

# 6. Build RetrievalQA Chain with Optimized Settings
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=False  # Skip returning sources to save processing time
)

# 7. Asynchronous Query Handling
async def query_qa_chain_async(query: str):
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, qa_chain.run, query)
    return response

# Query Example
async def main():
    query = "What is the content of the first email?"
    response = await query_qa_chain_async(query)
    print("Response:", response)

# Run the Async Query
asyncio.run(main())

"""
Detailed Improvements in the Code
Batch Processing with Async:

The query_qa_chain_async function runs the query in an asynchronous executor, allowing you to parallelize multiple user queries.
Reduced Retrieval Size (k):

Set k=3 for the retriever to fetch only the top 3 relevant results. Increase k if needed but keep it low to reduce computation.
Faster Embedding Model:

Switched to text-embedding-ada-002 for faster embedding generation without compromising too much on quality.
Smaller LLM:

Switched to gpt-3.5-turbo, which is faster and cheaper than gpt-4. If accuracy is critical, you can stick with gpt-4.
Concise Prompts:

The prompt is designed to include only the most relevant context ({context}), reducing token usage and LLM processing time.
Precomputed Embeddings:

If documents are static, embeddings can be precomputed during initialization and reused for all queries, avoiding repetitive computation.
"""

"""
Advanced Optimization: Use Approximate Nearest Neighbor (ANN)
If your vector store supports it (e.g., FAISS with HNSW), enable approximate nearest neighbor search for faster retrieval:
"""
from faiss import IndexHNSWFlat

# Create an HNSW index for faster retrieval
dimension = 1536  # Depends on the embedding model
index = IndexHNSWFlat(dimension, 32)  # 32 is the number of connections
vector_store = FAISS(index, embeddings.embed_query, "your_faiss_store")


"""
To reduce the token size of input for your Retrieval Augmented Generation (RAG) system, you can trim retrieved documents or summarize them before passing them to the language model. This ensures that your prompt stays within the model’s token limits, speeds up inference, and reduces cost.

Here’s how you can implement this using LangChain.

1. Trimming Retrieved Documents
Method: Restrict the number of tokens per retrieved document and prioritize the most relevant parts.

Implementation:
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Initialize Text Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Limit chunks to 500 characters
    chunk_overlap=50  # Add overlap for better context continuity
)

# Trimming Function
def trim_documents(documents, max_tokens=1500):
    """Trim the retrieved documents to fit within max_tokens."""
    trimmed_docs = []
    token_count = 0

    for doc in documents:
        # Split document into smaller chunks
        chunks = text_splitter.split_text(doc.page_content)

        for chunk in chunks:
            # Count tokens approximately (1 token ≈ 4 chars for English text)
            chunk_tokens = len(chunk) // 4
            if token_count + chunk_tokens <= max_tokens:
                trimmed_docs.append(chunk)
                token_count += chunk_tokens
            else:
                break  # Stop adding chunks if token limit is reached

    return " ".join(trimmed_docs)

# Usage in a Retrieval Chain
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
retrieved_docs = retriever.get_relevant_documents("Your query here")

# Trim documents before passing to the LLM
context = trim_documents(retrieved_docs)

# Define Prompt
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    You are a helpful assistant. Based on the context provided below, answer the user's question.

    Context: {context}

    Question: {question}
    Answer:
    """
)

# Pass to RetrievalQA Chain
llm = ChatOpenAI(model="gpt-4")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt_template},
    return_source_documents=False
)

response = qa_chain.run({"context": context, "question": "Your query here"})
print(response)


"""
2. Summarizing Retrieved Documents with a Summary Chain
Method: Use a summarization chain to condense each document into a shorter form before feeding it into the LLM.

Implementation:
"""

from langchain.chains import LLMChain, TransformChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate

# Define Summarization Prompt
summary_prompt = PromptTemplate(
    input_variables=["document"],
    template="""
    Summarize the following document concisely in one paragraph:
    {document}
    """
)

# Summarization Chain
summary_chain = LLMChain(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),  # Use a smaller, faster model for summarization
    prompt=summary_prompt
)

# Summarize Retrieved Documents
def summarize_documents(retrieved_docs):
    summaries = []
    for doc in retrieved_docs:
        summary = summary_chain.run({"document": doc.page_content})
        summaries.append(summary)
    return " ".join(summaries)  # Combine all summaries into a single context

# Retrieve and Summarize Documents
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
retrieved_docs = retriever.get_relevant_documents("Your query here")
summarized_context = summarize_documents(retrieved_docs)

# Define Prompt for QA
qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    You are a helpful assistant. Based on the summarized context provided below, answer the user's question.

    Summarized Context: {context}

    Question: {question}
    Answer:
    """
)

# QA Chain with Summarized Context
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4"),
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": qa_prompt},
    return_source_documents=False
)

response = qa_chain.run({"context": summarized_context, "question": "Your query here"})
print(response)

"""
3. Combining Both Trimming and Summarization
You can combine trimming and summarization for more efficient preprocessing:

Trim documents first to limit the number of tokens.
Summarize the trimmed chunks to condense information further.
"""
def preprocess_documents(retrieved_docs, max_tokens=1500):
    """Trim and summarize retrieved documents."""
    # Step 1: Trim Documents
    trimmed_context = trim_documents(retrieved_docs, max_tokens=max_tokens)

    # Step 2: Summarize the Trimmed Context
    summarized_context = summary_chain.run({"document": trimmed_context})
    return summarized_context

retriever = vector_store.as_retriever(search_kwargs={"k": 5})
retrieved_docs = retriever.get_relevant_documents("Your query here")

# Preprocess documents
final_context = preprocess_documents(retrieved_docs)

# Define QA Prompt
qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    You are a helpful assistant. Based on the condensed context provided below, answer the user's question.

    Condensed Context: {context}

    Question: {question}
    Answer:
    """
)

# QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4"),
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": qa_prompt},
    return_source_documents=False
)

response = qa_chain.run({"context": final_context, "question": "Your query here"})
print(response)



