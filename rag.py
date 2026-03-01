from config.db import qdrant_client
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from setup import setup_logging
import numpy as np
from typing import List
from model_registry import ModelRegistry

session = requests.Session()
retry = Retry(connect=3, backoff_factor=3)
adapter = HTTPAdapter(max_retries=retry)
session.mount("http://", adapter)
session.mount("https://", adapter)
logger = setup_logging()


class ArabicRagAnalyzer:
    def __init__(
        self,
        collection_all_case: str = None,
        collection_case: str = None,
        case_id: str = None,
        models_profile: any = None,
    ) -> None:
        self.collection_all_case = collection_all_case
        self.collection_case = collection_case
        self.case_id = case_id
        self.models_profile = models_profile
        embedding_obj = models_profile.get("embeddings", {})
        model_name = embedding_obj.get("name")
        self.embedding_client = ModelRegistry.get_model("embeddings", model_name=model_name)
        self.llamaClient = ModelRegistry.get_model("llama")

    def query_embedding_endpoint(self, prompt: str) -> List[List[float]]:
        """Get embeddings from local embeddings model"""
        logger.info("::Generating embeddings with local model::")
        try:
            embeddings = self.embedding_client.create_embeddings([prompt])
            normalized_embedding = self.normalize_embedding(embeddings[0])
            return normalized_embedding.tolist() if isinstance(normalized_embedding, np.ndarray) else normalized_embedding
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return None

    def normalize_embedding(self, embedding):
        return embedding / np.linalg.norm(embedding)

    def semantic_search(self, text, collection_name, limit, score_threshold):
        try:
            logger.info(f"Querying Hugging Face endpoint for embeddings... {text}")
            response = self.query_embedding_endpoint(text)
            logger.info(f"Response: {response}")
            if not response:
                return None
            results = qdrant_client.search(
                collection_name=collection_name,
                query_vector=response,
                limit=limit,
                score_threshold=score_threshold,
            )
            logger.info(f"Results for finding revelant points in the {collection_name} are:\n {results}")
 
            if not results:
                return None

            return results
        except Exception as e:
            logger.error(f"Error in Semantic Search: {str(e)}")
            return None

    def summarize_llama_messages(self, query, messages: list) -> None:
        """
        Summarize messages using the Llama 3.1:8b client.
        """
        try:
            messages_content_list = [
                f"{i+1}. {messages[i].payload.get("message")}"
                for i in range(len(messages))
            ]
            message_content_string = ""
            for message in messages_content_list:
                message_content_string += f"{message}\n"

            prompt = """You are a case investigator the three most relevant messages to the question from the case data are:
                    {context}
                    The user question is: {query}
                    Use the retrieved information to generate a concise and structured response.
                    Ensure the answer is fact-based and avoids unnecessary details.
                    Provide the answer in a clear, summarized format.
                    Give the answer in arabic only and do not output anything else other than the answer"""
            variables = {"query": query, "context": message_content_string}
            response = self.llamaClient.chat(prompt=prompt, variables=variables)
            if not response:
                return None
            return response
        except Exception as e:
            logger.error(f"Llama Text Generation Error: {str(e)}")
            return None

    def summarize_messages(self, query) -> None:
        """
        Fetch message by user query and return summary.
        """
        try:
            semantic_response = self.semantic_search(
                query=query,
                collection_name=f"case_{self.case_id}",
                limit=3,
                score_threshold=0.15,
            )
            if not semantic_response:
                return {
                    "success": False,
                    "message": "Failed to retrieve relevant messages.",
                }
            logger.info(f"Retrieved {len(semantic_response)} relevant messages.")

            # Extract mongo_ids from semantic_response
            mongo_ids = [
                item.payload.get("mongo_id")
                for item in semantic_response
                if item.payload and "mongo_id" in item.payload
            ]

            # summary = self.summarize_llm_messages(query, semantic_response)
            summary = self.summarize_llama_messages(query, semantic_response)

            if not summary:
                return {
                    "success": False,
                    "message": "Failed to generate summary.",
                    "mongo_ids": None,
                }
            return {
                "success": True,
                "summary": summary,
                "mongo_ids": mongo_ids,
            }
        except Exception as e:
            logger.error(f"Summarization error: {str(e)}")
            return {
                "success": False,
                "message": "Error in summarization.",
                "semantic_response": None,
            }
