import os
import logging
import uvicorn
import shutil
from fastapi import FastAPI, UploadFile, File, Form
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
UPLOAD_DIR = "uploads"
FAISS_INDEX_PATH = "faiss_index"
MODEL_PATH = "llama-2-7b.Q3_K_S.gguf"  # 🔥 Update with actual model path!

# Ensure upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize FastAPI
app = FastAPI()

# Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Global FAISS index
vector_store = None  

def load_faiss_index():
    global vector_store
    try:
        vector_store = FAISS.load_local(
            FAISS_INDEX_PATH, embedding_model, allow_dangerous_deserialization=True  
        )
        logger.info("✅ FAISS index loaded successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to load FAISS index: {e}")
        vector_store = None

# Load FAISS at startup
load_faiss_index()

# Load Llama model
try:
    llm = LlamaCpp(model_path=MODEL_PATH, max_tokens=256, temperature=0.2)
    logger.info("✅ Llama model loaded successfully!")
except Exception as e:
    logger.error(f"❌ Failed to load Llama model: {e}")
    llm = None

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file and update FAISS index.
    """
    pdf_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    logger.info(f"📄 PDF uploaded: {pdf_path}")

    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)

        global vector_store
        vector_store = FAISS.from_documents(texts, embedding_model)
        vector_store.save_local(FAISS_INDEX_PATH)

        load_faiss_index()

        return {"message": "✅ PDF uploaded and FAISS index created!"}
    except Exception as e:
        logger.error(f"❌ Error processing PDF: {e}")
        return {"error": "Failed to process PDF"}
@app.post("/query_pdf/")
async def query_pdf(query: str = Form(...)):
    if vector_store is None:
        return {"response": "⚠️ No PDF uploaded yet!"}

    try:
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})  
        retrieved_docs = retriever.get_relevant_documents(query)

        if not retrieved_docs:
            return {"response": "❌ No relevant information found."}

        extracted_text = "\n\n".join([doc.page_content.strip() for doc in retrieved_docs])[:700]

        llm_prompt = f"""
        Answer the question **only using the document text**.  
        If the answer is not found, say: "❌ The uploaded PDF does not contain relevant information."

        --- DOCUMENT TEXT ---
        {extracted_text}
        --- END OF DOCUMENT TEXT ---

        **Question:** {query}

        **Answer:**
        """
        
        response = llm.invoke(llm_prompt)
        
        print("🔹 RESPONSE:", response)  # Debugging

        return {"response": response.strip()}
    
    except Exception as e:
        logger.error(f"❌ Query error: {e}")
        return {"response": "Error occurred while querying."}
