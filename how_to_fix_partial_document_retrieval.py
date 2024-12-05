"""
Possible Reasons:
Indexing Issue:

Only 3 emails may have been correctly indexed into the PostgreSQL vector store.
The remaining 7 emails might not have been saved due to formatting, vector generation errors, or database insertion issues.
Retrieval Parameters:

The retriever might be set up to return only the top k results, with k=3 as the default.
This would limit the agent to only retrieve 3 emails regardless of the total available in the vector store.
Embeddings Similarity:

The embeddings for some emails may not closely match the query (e.g., "how many emails do you have"). This could cause only a subset of the documents to be retrieved.
Preprocessing Issues:

Some documents may have been improperly preprocessed or tokenized before being inserted, causing them to either fail embedding creation or not be properly searchable.
Query Misunderstanding:
"""

"""
2. Adjust Retrieval Parameters
Ensure the k parameter for retrieval is set to retrieve all documents if needed:
"""
retriever = vector_store.as_retriever(search_kwargs={"k": 10})


"""
3. Ensure Query Matches the Intent
Use a better prompt or logic for the agent to understand the intent of "how many emails do you have." For example, rewrite the prompt:
"""
from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["context"],
    template="You are a helpful assistant. Based on the provided context, answer this question: {context}"
)

"""
4. Test Embedding Quality
Verify that all emails are properly embedded and retrievable. For example:
"""
query_embedding = embeddings.embed_query("how many emails do you have")
results = vector_store.similarity_search_by_vector(query_embedding, k=10)

print("Retrieved documents:", results)
