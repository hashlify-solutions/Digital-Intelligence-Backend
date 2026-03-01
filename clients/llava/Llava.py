from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from PIL import Image
import base64
from io import BytesIO
import os
import logging

logger = logging.getLogger(__name__)

class Llava:
    def __init__(self, model_name="llava:latest"):
        """
        Initialize the LLaVA client with Ollama.

        Args:
            model_name (str): Name of the Ollama model (default: "llava:7b", "llava:v1.6", "llava:latest") which is the latest version.
        """
        self.llm = ChatOllama(model=model_name, temperature=0.1)

    def _image_to_base64(self, image_path) -> str:
        """
        Convert an image file to a base64-encoded string.

        Args:
            image_path (str): Path to the image file.

        Returns:
            str: Base64-encoded image string.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode != "RGB":
                img = img.convert("RGB")

            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return img_str

    def describe_image(self, image_path, question="What is in this image?") -> str:
        """
        Generate a description for the given image.

        Args:
            image_path (str): Path to the image file.
            question (str): Question about the image.

        Returns:
            str: Description of the image.
        """
        try:
            image_data = self._image_to_base64(image_path)
            message = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": question
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}"
                        }
                    }
                ]
            )
            response = self.llm.invoke([message])
            return response.content.strip()

        except Exception as e:
            logger.error(f"Error describing image with Llava: {str(e)}")
            return None


# if __name__ == "__main__":
#     client = Llava()
#     image_path = "./data/case_1_68a1e7ddf9c98f7327949684/detectors/dog_68a9d1ba01a532860d110953.jpg"
#     description = client.describe_image(image_path)
#     print("Description:", description)