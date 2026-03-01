from clients.llama.llama_v1 import LlamaClient
import json

def parse_user_query(user_query: str):
    """
    Use LLM to parse the user query into:
    - semantic_query: for embedding search
    - structured_filter: for MongoDB querying
    """
    prompt = """
    Given the following user question, do two things:
    1. Restate the question in a way that captures its semantic meaning for embedding search.
    2. Extract any structured filters (e.g., location, sentiment, date, user) as a JSON object for MongoDB querying.
    Return your answer as a JSON object with two keys: "semantic_query" and "structured_filter".
    User question: {user_query}
    """
    client = LlamaClient(prompt=prompt, variables={"user_query": user_query})
    response = client.chat()
    if isinstance(response, str):
        response = json.loads(response)
    return response