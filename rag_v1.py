from config.db import qdrant_client
from typing import Dict
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from setup import setup_logging
import numpy as np
import time
from utils.prompts import generate_english_prompt, generate_arabic_prompt, chunk_document, embed_and_store, retrieve_relevant_chunks
from clients.llama.llama_v1 import LlamaClient
from config.db import models_master_collection, processing_profiles_collection
from typing import List
from model_registry import ModelRegistry
from utils.llm_query_parser import parse_user_query
import asyncio
from concurrent.futures import ThreadPoolExecutor
from config.settings import settings

session = requests.Session()
retry = Retry(connect=3, backoff_factor=3)
adapter = HTTPAdapter(max_retries=retry)
session.mount('http://', adapter)
session.mount('https://', adapter)
logger = setup_logging()


class ArabicRagAnalyzer:
    def __init__(self, collection_all_case: str= None, collection_case: str= None, case_id: str= None, models_profile: any= None, use_parallel_processing: bool = True) -> None:
        self.collection_all_case = collection_all_case
        self.collection_case = collection_case
        self.case_id = case_id
        self.models_profile = models_profile
        self.use_parallel_processing = use_parallel_processing
        
        embedding_obj = models_profile.get("embeddings", {})
        model_name = embedding_obj.get("name")
        
        # Initialize embedding client with settings from centralized config
        compute = settings.compute_config
        self.embedding_client = ModelRegistry.get_model("embeddings", model_name=model_name)
        if self.use_parallel_processing:
            # Update embedding client with optimized parameters
            if hasattr(self.embedding_client, 'batch_size'):
                self.embedding_client.batch_size = compute.get("embedding_batch_size", 64)
            if hasattr(self.embedding_client, 'max_workers'):
                self.embedding_client.max_workers = compute.get("max_workers", 4)
        
        self.chat_history = []  # Initialize empty chat history
        self.conversation_state = {
            "current_topic": None,
            "follow_up_questions": [],
            "context_window": 5,  # Number of previous exchanges to consider
            "last_entities": [],
            "last_topics": []
        }
        
        # Initialize thread pool for parallel processing
        if self.use_parallel_processing:
            self.executor = ThreadPoolExecutor(max_workers=compute.get("max_workers", 4))
        else:
            self.executor = None
                
    async def query_embedding_endpoint_async(self, prompt: str) -> List[List[float]]:
        """Get embeddings from local embeddings model asynchronously"""
        logger.info("::Generating embeddings with local model (async)::")
        try:
            if self.use_parallel_processing and hasattr(self.embedding_client, 'create_embeddings_async'):
                embeddings = await self.embedding_client.create_embeddings_async([prompt])
            else:
                embeddings = self.embedding_client.create_embeddings([prompt])
            return embeddings
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return None
        
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
        
    async def query_embedding_endpoint_dict_async(self, data: Dict[str, List]) -> Dict[str, List[List[float]]]:
        """Get embeddings for multiple columns from local embeddings model asynchronously"""
        logger.info("::Generating embeddings for multiple columns with local model (async)::")
        try:
            if self.use_parallel_processing and hasattr(self.embedding_client, 'create_embeddings_from_dict_async'):
                return await self.embedding_client.create_embeddings(data.get("Preview Text", ""))
            else:
                return self.embedding_client.create_embeddings(data.get("Preview Text", ""))
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return None
        
    # def query_embedding_endpoint_dict(self, data: Dict[str, List]) -> Dict[str, List[List[float]]]:
    #     """Get embeddings for multiple columns from local embeddings model"""
    #     logger.info("::Generating embeddings for multiple columns with local model::")
    #     try:
    #         # Convert dict of lists to list of dicts for batching
    #         keys = list(data.keys())
    #         batch_list = [dict(zip(keys, t)) for t in zip(*data.values())]
    #         results = await async_generate_embeddings(batch_list, endpoint)
    #         # Ensure the result is a dict mapping keys to embeddings
    #         if isinstance(results, dict):
    #             return results
    #         elif isinstance(results, list):
    #             # If a list, convert to dict with keys
    #             return {k: v for k, v in zip(keys, results)}
    #         else:
    #             logger.error(f"Unexpected result type from async_generate_embeddings: {type(results)}")
    #             return {}
    #     except Exception as e:
    #         logger.error(f"Async embedding generation (dict) failed: {e}")
    #         return None
        
    def normalize_embedding(self, embedding):
        return embedding / np.linalg.norm(embedding)
    
    async def semantic_search_async(self, text, limit=20):
        """Enhanced semantic search with async embedding generation"""
        try:
            logger.info(f"Querying embeddings endpoint for embeddings (async)... {text}")
            response = await self.query_embedding_endpoint_async(text)
            logger.info(f"Response: {response}")
            if not response:
                return None
            embeddings = self.normalize_embedding(response[0])
            
            # Increase limit for better context
            results = qdrant_client.search(
                collection_name=f"case_{self.case_id}",
                query_vector=embeddings,
                limit=limit * 2,  # Get more matches for better context
                score_threshold=0.12  # Lower threshold to get more relevant results
            )
            
            # Filter and sort results by relevance
            filtered_results = [r for r in results if r.score > 0.15]
            sorted_results = sorted(filtered_results, key=lambda x: x.score, reverse=True)[:limit]
            
            logger.info(f"Filtered Results: {sorted_results}")
            if not sorted_results:
                return None
                
            return sorted_results
        except Exception as e:
            logger.error(f"Error in Semantic Search (async): {str(e)}")
            return None
    
    # def semantic_search(self, text, limit=20):
    #     try:
    #         logger.info(f"Querying Hugging Face endpoint for embeddings... {text}")
    #         response = await self.query_embedding_endpoint([text], self.models_profile.get("embeddings", {}).get("endpoint", ""))
    #         logger.info(f"Response: {response}")
    #         if not response:
    #             return None
    #         embeddings = self.normalize_embedding(response[0])

    #         # Use async Qdrant search
    #         results = await async_qdrant_search(
    #             collection_name=f"case_{self.case_id}",
    #             query_vector=embeddings,
    #             limit=limit * 2,
    #             score_threshold=0.12
    #         )

    #         # Filter and sort results by relevance
    #         filtered_results = [r for r in results if r.score > 0.15]
    #         sorted_results = sorted(filtered_results, key=lambda x: x.score, reverse=True)[:limit]

    #         logger.info(f"Filtered Results: {sorted_results}")
    #         if not sorted_results:
    #             return None

    #         return sorted_results
    #     except Exception as e:
    #         logger.error(f"Error in Semantic Search: {str(e)}")
    #         return None

    def is_greeting(self, query: str) -> bool:
        """
        Check if the query is a greeting in Arabic.
        """
        # Common Arabic greetings and their variations
        greetings = [
            "مرحبا", "اهلا", "السلام", "هاي", "هلا", "شكرا", "مشكور", "مع السلامة", "باي",
            "مرحبتين", "اهلين", "هلا وغلا", "هلا والله", "هلا وسهلا", "هلا وسهلا وغلا",
            "شكراً", "شكرا جزيلا", "شكراً جزيلاً", "شكراً لك", "شكراً لكم",
            "مع السلامة", "الله يسلمك", "الله يسلمكم", "الله يخليك", "الله يخليكم",
            "باي", "مع السلامة", "الله يسلمك", "الله يسلمكم", "الله يخليك", "الله يخليكم",
            "صباح الخير", "مساء الخير", "تصبح على خير", "مساء النور", "صباح النور",
            "كيف حالك", "كيفك", "كيف الحال", "كيف حالكم", "كيفكم"
        ]
        
        # Remove any extra spaces and normalize the query
        query = " ".join(query.split())
        
        # Check if any greeting is in the query
        return any(greeting in query for greeting in greetings)

    def update_conversation_state(self, query: str, response: str, mongo_ids: list = None):
        """Update the conversation state with new information."""
        # Add to chat history
        self.chat_history.append({
            "query": query,
            "response": response,
            "mongo_ids": mongo_ids or [],
            "timestamp": time.time()
        })
        
        # Keep only the last N exchanges based on context window
        if len(self.chat_history) > self.conversation_state["context_window"]:
            self.chat_history = self.chat_history[-self.conversation_state["context_window"]:]
        
        # Extract and update entities and topics from the response
        if mongo_ids:
            try:
                # Get the most recent messages for context
                recent_messages = [msg for msg in self.chat_history[-3:] if msg.get("mongo_ids")]
                if recent_messages:
                    # Update conversation state with new context
                    self.conversation_state["last_entities"] = list(set(
                        entity for msg in recent_messages 
                        for entity in msg.get("entities", [])
                    ))
                    self.conversation_state["last_topics"] = list(set(
                        topic for msg in recent_messages 
                        for topic in msg.get("topics", [])
                    ))
            except Exception as e:
                logger.error(f"Error updating conversation state: {e}")

    def get_conversation_context(self) -> str:
        """Generate a context string from recent conversation history."""
        if not self.chat_history:
            return ""
            
        context = "\nالمحادثة السابقة:\n"
        for chat in self.chat_history[-self.conversation_state["context_window"]:]:
            context += f"المستخدم: {chat['query']}\nالمساعد: {chat['response']}\n"
            
        if self.conversation_state["last_entities"]:
            context += f"\nالمواضيع السابقة: {', '.join(self.conversation_state['last_topics'])}\n"
            context += f"الكيانات المذكورة: {', '.join(self.conversation_state['last_entities'])}\n"
            
        return context

    async def summarize_llama_messages(self, query, messages: list) -> str:
        """
        Async: Summarize messages using the LLM with enhanced RAG capabilities.
        """
        try:
            # Check if this is a greeting or general conversation
            if self.is_greeting(query):
                prompt = f"""أنت مساعد ذكي ودود مثل ChatGPT.
المستخدم قال: {query}
رد على المستخدم بشكل إنساني وودود.
إذا كان تحية، رد بتحية مناسبة.
إذا كان شكراً، رد بطريقة مهذبة.
إذا كان وداعاً، رد بتحية وداع مناسبة.
كن ودوداً ومهذباً في ردك.
أجب باللغة العربية فقط.
"""
                client = LlamaClient(prompt=prompt, variables={}, prompt_engineering=self.models_profile.get('llama', {}).get('prompt_engineering', ''))
                response = client.chat()
                return response or "مرحباً! كيف يمكنني مساعدتك اليوم؟"

            # For non-greeting queries, proceed with enhanced RAG processing
            messages_content_list = []
            for i, msg in enumerate(messages):
                payload = msg.payload if hasattr(msg, 'payload') else (msg or {})
                message_content = (payload.get('message') if isinstance(payload, dict) else '') or ''
                meta_from = (payload.get('from') if isinstance(payload, dict) else None) or 'N/A'
                meta_to = (payload.get('to') if isinstance(payload, dict) else None) or 'N/A'
                meta_date = (payload.get('date') if isinstance(payload, dict) else None) or 'N/A'
                meta_case = (payload.get('case_name') if isinstance(payload, dict) else None) or 'N/A'
                metadata_text = f"from: {meta_from} | to: {meta_to} | date: {meta_date} | case: {meta_case}"
                messages_content_list.append(f"{i+1}. {message_content}\nMetadata: {metadata_text}")

            message_content_string = "\n\n".join(messages_content_list)
            chat_history_context = self.get_conversation_context()
            prompt = f"""أنت مساعد ذكي ودود مثل ChatGPT.
مهمتك الرئيسية هي تقديم إجابات دقيقة ومفصلة بناءً على المعلومات المسترجعة من القضية.

المعلومات المسترجعة من القضية:
{message_content_string}

{chat_history_context}

سؤال المستخدم: {query}

قواعد الرد:
1. ركز على المعلومات المسترجعة من القضية في إجابتك
2. استخدم التفاصيل والبيانات الواردة في المعلومات المسترجعة
3. إذا كان السؤال يتعلق بمعلومات محددة، ابحث عنها في المعلومات المسترجعة
4. إذا لم تجد المعلومات المطلوبة في النتائج المسترجعة، قل ذلك بوضوح
5. استخدم سياق المحادثة السابقة لتقديم رد أكثر تفاعلية وترابطاً
6. كن دقيقاً في نقل المعلومات من النتائج المسترجعة
7. إذا كان السؤال متابعة لسؤال سابق، استخدم المعلومات من المحادثة السابقة
8. أجب باللغة العربية فقط
9. كن ودوداً ومهذباً في ردك
10. إذا كان السؤال عاماً، أجب بشكل مفيد وودود
"""
            client = LlamaClient(prompt=prompt, variables={}, prompt_engineering=self.models_profile.get('llama', {}).get('prompt_engineering', ''))
            response = client.chat()
            return response
        except Exception as e:
            logger.error(f"Llama Text Generation Error: {str(e)}")
            return None

    async def summarize_messages_async(self, query) -> None:
        """
        Async version of summarize_messages with parallel processing support
        """
        try:
            # Check if this is a greeting or general conversation
            if self.is_greeting(query):
                prompt = f"""أنت مساعد ذكي ودود مثل ChatGPT.
                المستخدم قال: {query}
                رد على المستخدم بشكل إنساني وودود.
                إذا كان تحية، رد بتحية مناسبة.
                إذا كان شكراً، رد بطريقة مهذبة.
                إذا كان وداعاً، رد بتحية وداع مناسبة.
                كن ودوداً ومهذباً في ردك.
                أجب باللغة العربية فقط.
                """
                variables = {"query": query}
                llamaClient = LlamaClient(prompt=prompt, variables=variables, prompt_engineering=self.models_profile.get('llama', {}).get('prompt_engineering', ''))
                response = llamaClient.chat()
                
                # Update conversation state
                self.update_conversation_state(query, response)

                return {
                    "success": True,
                    "summary": response,
                    "mongo_ids": []
                }

            # For non-greeting queries, proceed with enhanced semantic search
            semantic_response = await self.semantic_search_async(query, limit=20)
            if not semantic_response:
                # If no relevant messages found, generate a more contextual response
                response = await self.fallback_chatgpt_style_response(query)
                
                # Update conversation state
                self.update_conversation_state(query, response)
                
                return {
                    "success": True,
                    "summary": response,
                    "mongo_ids": []
                }

            # Extract mongo_ids and additional metadata from semantic_response
            mongo_ids = []
            metadata = []
            for item in semantic_response:
                if item.payload and "mongo_id" in item.payload:
                    mongo_ids.append(item.payload.get("mongo_id"))
                    metadata.append({
                        "score": item.score,
                        "from": item.payload.get("from"),
                        "to": item.payload.get("to"),
                        "date": item.payload.get("date")
                    })
            
            # Generate summary with enhanced context
            summary = await self.summarize_llama_messages(query, semantic_response)

            if not summary:
                # If summary generation fails, use fallback response
                response = await self.fallback_chatgpt_style_response(query)
                
                # Update conversation state
                self.update_conversation_state(query, response)
                
                return {
                    "success": True,
                    "summary": response,
                    "mongo_ids": []
                }

            # Update conversation state with successful response
            self.update_conversation_state(query, summary, mongo_ids)

            return {
                "success": True,
                "summary": summary,
                "mongo_ids": mongo_ids,
                "metadata": metadata
            }
        except Exception as e:
            logger.error(f"Summarization error (async): {str(e)}")
            # Use fallback response in case of errors
            response = await self.fallback_chatgpt_style_response(query)
            
            # Update conversation state with fallback response
            self.update_conversation_state(query, response)
            
            return {
                "success": True,
                "summary": response,
                "mongo_ids": []
            }
    
    async def dual_search_and_answer(self, user_query: str, top_k: int = 3):
        """
        Async: Perform dual search (semantic + MongoDB) and answer using LLM.
        """
        parsed = parse_user_query(user_query)
        semantic_query = parsed["semantic_query"]
        structured_filter = parsed["structured_filter"]

        # 2. Semantic search (Qdrant, async)
        semantic_results = await self.semantic_search_async(semantic_query, limit=top_k) or []

        # 3. Structured search (MongoDB, async)
        try:
            cursor = self.collection_case.find(structured_filter).limit(top_k)
            structured_results = await cursor.to_list(length=top_k)
        except Exception as e:
            logger.error(f"Structured search failed: {e}")
            structured_results = []

        # 4. Merge & deduplicate
        all_results = {}
        for r in semantic_results:
            mongo_id = r.payload.get("mongo_id") if hasattr(r, "payload") else (r.get("mongo_id") if isinstance(r, dict) else None)
            if mongo_id:
                all_results[mongo_id] = r
        for doc in structured_results:
            mongo_id = str(doc.get("_id"))
            all_results[mongo_id] = doc
        merged_results = list(all_results.values())

        # 5. LLM for answer only (no chart)
        answer = await self.fallback_chatgpt_style_response(user_query)

        return {
            "answer": answer,
            "results": merged_results
        }

    async def fallback_chatgpt_style_response(self, user_query):
        """
        Async: Generate a more natural and contextual response when no RAG data is found.
        """
        chat_history_context = ""
        if self.chat_history:
            chat_history_context = "\nالمحادثة السابقة:\n"
            for chat in self.chat_history[-3:]:
                chat_history_context += f"المستخدم: {chat['query']}\nالمساعد: {chat['response']}\n"

        prompt = f"""أنت مساعد ذكي ودود مثل ChatGPT.
        المستخدم سأل: {user_query}
        
        {chat_history_context}
        
        قواعد الرد:
        1. إذا كان السؤال عاماً، أجب بشكل مفيد وودود
        2. إذا كان السؤال خارج نطاق معرفتك، قل ذلك بأدب واقترح بدائل
        3. إذا كان السؤال عن معلومات محددة، اشرح أنك لا تملك هذه المعلومات المحددة
        4. حاول دائماً أن تكون مفيداً حتى لو لم يكن لديك المعلومات المطلوبة
        5. أجب باللغة العربية فقط
        6. كن طبيعياً في ردك وكأنك تتحدث مع صديق
        7. إذا كان السؤال عن القضية، اشرح أنك لا تملك معلومات كافية عن هذه القضية المحددة
        8. اقترح أسئلة بديلة أو مواضيع ذات صلة يمكنك المساعدة فيها
        """
        client = LlamaClient(prompt=prompt, variables={}, prompt_engineering=self.models_profile.get('llama', {}).get('prompt_engineering', ''))
        response = client.chat()
        return response
    
    def chunk_and_embed_document(self, text: str, is_html: bool = False, chunk_size: int = 500, chunk_overlap: int = 100):
        """
        Chunk a long document and embed the chunks using the configured embedding model.
        Returns the vectorstore (FAISS) and the list of chunks.
        """
        chunks = chunk_document(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap, is_html=is_html)
        if not chunks:
            logger.warning("No chunks generated from document.")
            return None, []
        # Use the embedding client if compatible, else fallback to OpenAIEmbeddings
        try:
            from langchain_community.embeddings import OpenAIEmbeddings
            embedding_model = getattr(self, 'embedding_client', None) or OpenAIEmbeddings()
        except Exception:
            from langchain_community.embeddings import OpenAIEmbeddings
            embedding_model = OpenAIEmbeddings()
        vectorstore = embed_and_store(chunks, embedding_model)
        return vectorstore, chunks

    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'executor') and self.executor:
            self.executor.shutdown(wait=True)
    