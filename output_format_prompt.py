from langchain.prompts import PromptTemplate

# Define a prompt template for structured and bolded responses
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    You are a helpful assistant that provides structured and professional answers. Respond to the user's question based on the given context in the following format:

    - Use **bold subtitles** for each main section.
    - Use bullet points or numbered lists for detailed explanations when appropriate.
    - Ensure the response is clear, concise, and easy to read.

    Context: {context}

    Question: {question}

    Your well-structured answer:
    """
)

# Example usage
print(prompt.format(context="The user provided context here.", question="What are the benefits of AI?"))
