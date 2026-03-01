def generate_english_prompt(query: str, messages: list) -> str:
    """
    Generate a prompt for the English GPT-3 model.
    """
    # Check if this is a greeting or general conversation
    is_greeting = any(word in query.lower() for word in [
        "مرحبا", "اهلا", "السلام", "هاي", "هلا", "شكرا", "مشكور", "مع السلامة", "باي",
        "مرحبتين", "اهلين", "هلا وغلا", "هلا والله", "هلا وسهلا", "هلا وسهلا وغلا",
        "شكراً", "شكرا جزيلا", "شكراً جزيلاً", "شكراً لك", "شكراً لكم",
        "مع السلامة", "الله يسلمك", "الله يسلمكم", "الله يخليك", "الله يخليكم",
        "باي", "مع السلامة", "الله يسلمك", "الله يسلمكم", "الله يخليك", "الله يخليكم",
        "صباح الخير", "مساء الخير", "تصبح على خير", "مساء النور", "صباح النور",
        "كيف حالك", "كيفك", "كيف الحال", "كيف حالكم", "كيفكم"
    ])
    
    if is_greeting:
        return f"""أنت مساعد ذكي ودود مثل ChatGPT.
        المستخدم قال: {query}
        رد على المستخدم بشكل إنساني وودود.
        إذا كان تحية، رد بتحية مناسبة.
        إذا كان شكراً، رد بطريقة مهذبة.
        إذا كان وداعاً، رد بتحية وداع مناسبة.
        كن ودوداً ومهذباً في ردك.
        أجب باللغة العربية فقط.
        """
    
    context = "\n".join([f"{i+1}) {msg.payload.get('message', '')}" for i, msg in enumerate(messages)])
    prompt = f"""أنت مساعد ذكي ودود مثل ChatGPT.
    إذا كان السؤال عاماً أو لا يتعلق بالمعلومات المسترجعة، أجب بشكل عام وودود.
    إذا كان السؤال يتعلق بالمعلومات المسترجعة، استخدمها في إجابتك.
    
    المعلومات المسترجعة من القضية:
    {context}
    
    سؤال المستخدم: {query}
    
    قواعد الرد:
    1. إذا كان السؤال عاماً أو تحية، رد بشكل ودود ومهذب
    2. إذا كان السؤال عن معلومات القضية، استخدم المعلومات المسترجعة
    3. إذا لم تجد معلومات كافية، أجب بشكل عام وودود
    4. كن دائماً مهذباً ومفيداً في ردك
    5. أجب باللغة العربية فقط
    """
    return prompt
  
def generate_arabic_prompt(query: str, messages: list) -> str:
    """
    Generate a prompt for the Arabic GPT-3 model.
    This is now identical to the English prompt since we're handling Arabic text.
    """
    return generate_english_prompt(query, messages)

# --- Chunking and Embedding Utilities ---
import re
from typing import List, Optional
from bs4 import BeautifulSoup

from langchain_text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings  # or HuggingFaceEmbeddings, etc.
from langchain_community.vectorstores import FAISS

def clean_text(text: str, is_html: bool = False) -> str:
    """
    Clean and preprocess the document (plain text or HTML).
    """
    if not text or not isinstance(text, str):
        return ""
    if is_html:
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text(separator=" ")
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def chunk_document(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    is_html: bool = False
) -> List[str]:
    """
    Split text into overlapping, semantically meaningful chunks.
    """
    cleaned = clean_text(text, is_html)
    if not cleaned:
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,  # Use len for characters; for tokens, use tiktoken or similar
        separators=["\n\n", "\n", ".", "!", "?", "،", "؛", " "]
    )
    chunks = splitter.split_text(cleaned)
    return [chunk.strip() for chunk in chunks if chunk.strip()]

def embed_and_store(
    chunks: List[str],
    embedding_model: Optional[object] = None
) -> FAISS:
    """
    Embed chunks and store in a FAISS vector store.
    """
    if not embedding_model:
        embedding_model = OpenAIEmbeddings()  # Or HuggingFaceEmbeddings(), etc.
    vectorstore = FAISS.from_texts(chunks, embedding_model)
    return vectorstore

def retrieve_relevant_chunks(
    query: str,
    vectorstore: FAISS,
    embedding_model: Optional[object] = None,
    k: int = 5
) -> List[str]:
    """
    Retrieve top-k relevant chunks for a query from the vector store.
    """
    if not embedding_model:
        embedding_model = OpenAIEmbeddings()
    docs_and_scores = vectorstore.similarity_search_with_score(query, k=k)
    return [doc.page_content for doc, score in docs_and_scores]

# --- End of chunking/embedding utilities ---