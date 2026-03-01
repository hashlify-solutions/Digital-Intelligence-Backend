from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json
import re
import ast
import logging

logger = logging.getLogger(__name__)


class Llama:
    def __init__(self) -> None:
        pass

    def _safe_parse_json(self, response: str, fallback_value=None):
        """
        Safely parse JSON response from LLM with fallback handling.

        Args:
            response: Raw response string from LLM
            fallback_value: Value to return if parsing fails

        Returns:
            Parsed JSON object or fallback_value
        """
        try:
            # First, try to parse the response as-is
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.warning(f"Initial JSON parsing failed: {e}")

        # Try to fix common LLM issues: single quotes instead of double quotes
        try:
            # Replace single quotes with double quotes for JSON compatibility
            fixed_response = re.sub(
                r"(?<=[:\[,\s])'([^']*)'(?=[,\]\}:\s]|$)", r'"\1"', response
            )
            return json.loads(fixed_response)
        except json.JSONDecodeError:
            pass

        # Try using ast.literal_eval for Python-style dicts/lists
        try:
            return ast.literal_eval(response)
        except (ValueError, SyntaxError):
            pass

        # Try to extract JSON from the response using regex
        try:
            # Look for JSON array pattern
            array_match = re.search(r"\[.*?\]", response, re.DOTALL)
            if array_match:
                extracted = array_match.group()
                # Try fixing quotes on extracted content too
                try:
                    return json.loads(extracted)
                except json.JSONDecodeError:
                    try:
                        return ast.literal_eval(extracted)
                    except (ValueError, SyntaxError):
                        pass

            # Look for JSON object pattern
            object_match = re.search(r"\{.*\}", response, re.DOTALL)
            if object_match:
                extracted = object_match.group()
                try:
                    return json.loads(extracted)
                except json.JSONDecodeError:
                    try:
                        return ast.literal_eval(extracted)
                    except (ValueError, SyntaxError):
                        pass

        except (json.JSONDecodeError, ValueError, SyntaxError) as e2:
            logger.warning(f"Regex-extracted JSON parsing also failed: {e2}")

        # Log the problematic response for debugging
        logger.error(
            f"Could not parse JSON from LLM response. Response was: {response}"
        )

        # Return fallback value
        return fallback_value

    def chat(self, prompt: str, variables: dict) -> str:
        prompt = PromptTemplate(
            template=prompt,
            input_variables=list(variables.keys()),
        )
        llm = ChatOllama(
            model="llama3.1:8b",
            temperature=0,
        )
        llm_chain = prompt | llm | StrOutputParser()
        answer = llm_chain.invoke(variables)
        return answer

    def extract_entities(self, preview_text: str):
        prompt = """Extract entities (names, locations, organizations, and other key information) from the following text:
        
        {preview_text}

        Return your response as a JSON string in exactly this format:
        [
            'entity1',
            'entity2',
            'entity3',
            ...
        ]
        Return ONLY a valid JSON array with no additional text, comments, or explanations.
        If no entities are found, return "None".       
        """

        response = self.chat(prompt=prompt, variables={"preview_text": preview_text})

        # If the model returns "None" or an empty response, return an empty list
        if "none" in response.lower():
            return []

        # Safely parse the JSON response
        entities = self._safe_parse_json(response, fallback_value=[])

        # Ensure we return a list
        if not isinstance(entities, list):
            logger.warning(
                f"Expected list from extract_entities, got {type(entities)}. Converting to list."
            )
            return []

        return entities

    def classify_entities(self, entities, categories):
        prompt = """Classify the given entities into one of the following categories:

            Entities:
            {entities}

            Categories:
            {categories}

            Return your response as a JSON string in exactly this format:
            
            {{
                "category1": ["entity1", "entity2", "entity3"],
                "category2": ["entity4", "entity5", "entity6"],
                ...
            }}

            Return ONLY a valid JSON array with no additional text, comments, or explanations.
            """

        response = self.chat(
            prompt=prompt, variables={"entities": entities, "categories": categories}
        )

        # Safely parse the JSON response
        classification = self._safe_parse_json(response, fallback_value={})

        # Ensure we return a dictionary
        if not isinstance(classification, dict):
            logger.warning(
                f"Expected dict from classify_entities, got {type(classification)}. Returning empty dict."
            )
            return {}

        return classification