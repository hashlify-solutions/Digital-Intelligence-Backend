from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from numpy import isin
from setup import setup_logging
from typing import Optional, Dict, Any, List
from config.settings import settings

logger = setup_logging()

class LlamaClient:
    # Default basic parameters
    DEFAULT_BASIC_PARAMS = {
        "temperature": 0.6,
        "num_ctx": 2048,
        "num_tokens": 512
    }

    # Default advanced parameters
    DEFAULT_ADVANCED_PARAMS = {
        "num_layers": 12,
        "hidden_size": 768,
        "activation_function": "GELU",
        "dropout_rate": 0.1,
        "attention_heads": 12,
        "vocab_size": 30522,
        "max_sequence_length": 128,
        "learning_rate": 3e-5,
        "optimizer": "AdamW",
        "batch_size": 32,
        "epochs": 5,
        "loss_function": "CrossEntropy",
        "weight_decay": 0.01,
        "gradient_clipping": 1.0,
        "early_stopping": True,
        "validation_split": 0.2,
        "learning_rate_scheduler": "linear_decay",
        "seed": 42,
        "log_interval": 50,
        "save_checkpoint": "every_epoch"
    }

    def __init__(
        self, 
        prompt: str = "", 
        variables: dict = {},
        basic_params: Optional[Dict[str, Any]] = None,
        advanced_params: Optional[Dict[str, Any]] = None,
        prompt_engineering: Optional[str] = None
    ) -> None:
        self.prompt = prompt
        self.variables = variables
        self.basic_params = {**self.DEFAULT_BASIC_PARAMS, **(basic_params or {})}
        self.advanced_params = {**self.DEFAULT_ADVANCED_PARAMS, **(advanced_params or {})}
        self.prompt_engineering = prompt_engineering
        self.timeout = settings.effective_llama_timeout

    def chat(self) -> str:
        try:
            final_prompt = self.prompt
            if self.prompt_engineering and self.prompt_engineering.strip():
                final_prompt = self.prompt_engineering.strip() + "\n" + self.prompt
            prompt = PromptTemplate(
                template=final_prompt,
                input_variables=list(self.variables.keys()),
            )
            
            model_params = {**self.basic_params, **self.advanced_params}
            
            llm = ChatOllama(
                model="llama3.1:8b",
                timeout=self.timeout,
                **model_params
            )
            
            llm_chain = prompt | llm | StrOutputParser()
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(llm_chain.invoke, self.variables)
                try:
                    answer = future.result(timeout=self.timeout)
                except FuturesTimeoutError:
                    logger.error(f"LlamaClient chat timed out after {self.timeout}s")
                    future.cancel()
                    return "عذراً، انتهت مهلة الاستجابة. يرجى المحاولة مرة أخرى."
            
            if not answer or len(answer.strip()) == 0:
                logger.warning("Empty response from LLM")
                return "عذراً، لم أتمكن من توليد رد مناسب. هل يمكنك إعادة صياغة سؤالك؟"
                
            return answer.strip()
            
        except Exception as e:
            logger.error(f"Error in LlamaClient: {str(e)}")
            return "عذراً، حدث خطأ في معالجة طلبك. يرجى المحاولة مرة أخرى."
    
    def _invoke_with_timeout(self, llm_chain, variables: dict) -> Optional[str]:
        """Invoke an LLM chain with a Python-level timeout safety net."""
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(llm_chain.invoke, variables)
            try:
                return future.result(timeout=self.timeout)
            except FuturesTimeoutError:
                logger.error(f"LlamaClient validation timed out after {self.timeout}s")
                future.cancel()
                return None

    def validate_classification_result(self, text: str, all_classes: List[str], classfication_result: Dict[str, Any]) -> Optional[float]:
        """Validate the classification result for the given text against all classes and return a score out of 10"""
        try:
            with open("prompts/classfication_validation.txt", "r") as file:
                classfication_validation_prompt = file.read()
                
            prompt = PromptTemplate(
                template=classfication_validation_prompt,
                input_variables=["text", "all_classes", "classification_label"]
            )
            
            llm = ChatOllama(
                model="llama3.1:8b",
                timeout=self.timeout,
                **self.basic_params
            )
            
            llm_chain = prompt | llm | StrOutputParser()
            answer = self._invoke_with_timeout(llm_chain, {"text": text, "all_classes": all_classes, "classification_label": classfication_result})
            if answer is None:
                return None
            
            return float(answer.strip())
        except ValueError as e:
            logger.warning(f"Invalid classification validation response from LLM: {e}")
            return None
        except Exception as e:
            logger.error(f"Error validating classification result: {e}")
            return None
    
    def validate_toxicity_result(self, text: str, toxicity_score: float) -> Optional[float]:
        """Validate the toxicity result for the given text and return a score out of 10"""
        try:
            with open("prompts/toxicity_validation.txt", "r") as file:
                toxicity_validation_prompt = file.read()
                
            prompt = PromptTemplate(
                template=toxicity_validation_prompt,
                input_variables=["text", "toxicity_score"]
            )
            
            llm = ChatOllama(
                model="llama3.1:8b",
                timeout=self.timeout,
                **self.basic_params
            )
            
            llm_chain = prompt | llm | StrOutputParser()
            answer = self._invoke_with_timeout(llm_chain, {"text": text, "toxicity_score": toxicity_score})
            if answer is None:
                return None
            
            return float(answer.strip())
        except ValueError as e:
            logger.warning(f"Invalid toxicity validation response from LLM: {e}")
            return None
        except Exception as e:
            logger.error(f"Error validating toxicity result: {e}")
            return None
    
    def validate_emotion_result(self, text: str, all_emotions: List[str], emotion_result: Dict[str, Any]) -> Optional[float]:
        """Validate the emotion result for the given text and return a score out of 10"""
        try:
            with open("prompts/emotion_validation.txt", "r") as file:
                emotion_validation_prompt = file.read()
                
            prompt = PromptTemplate(
                template=emotion_validation_prompt,
                input_variables=["text", "all_emotions", "emotion_label"]
            )
            
            llm = ChatOllama(
                model="llama3.1:8b",
                timeout=self.timeout,
                **self.basic_params
            )
            
            llm_chain = prompt | llm | StrOutputParser()
            answer = self._invoke_with_timeout(llm_chain, {"text": text, "all_emotions": all_emotions, "emotion_label": emotion_result})
            if answer is None:
                return None
            
            return float(answer.strip())
        except ValueError as e:
            logger.warning(f"Invalid emotion validation response from LLM: {e}")
            return None
        except Exception as e:
            logger.error(f"Error validating emotion result: {e}")
            return None