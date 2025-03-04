📄 PDF Chatbot Application



🚀 Overview


This PDF Chatbot Application allows users to upload PDFs and ask questions based on the document content. It leverages Natural Language Processing (NLP) with FAISS for vector search, Llama 2 for generating responses, and a FastAPI backend with a Streamlit UI.

🛠️ Features


✅ Upload PDFs and process text

✅ Retrieve relevant document content using FAISS

✅ Answer user queries using Llama 2

✅ Interactive UI built with Streamlit

✅ Deployable with Docker


⚙️ Tech Stack


FastAPI (Backend)

Streamlit (Frontend)

FAISS (Vector Search)

Llama 2 (LLM for answering questions)

HuggingFace Sentence Transformers (Embeddings)

PyMuPDF / PDFMiner (PDF Text Extraction)

Docker (Deployment)


🔧 Installation


1️⃣ Clone the Repository

git clone https://github.com/Gnanapriya2000/pdfchatbotapplication.git

cd pdf-chatbot

2️⃣ Install Dependencies

pip install -r requirements.txt  

🏗️ Usage


1️⃣ Start the FastAPI Backend

uvicorn app:app --host 0.0.0.0 --port 8009

2️⃣ Run the Streamlit Frontend

streamlit run app.py

🔗 Future Improvements

🚀 Multi-document support

🚀 Model fine-tuning for better answers

🚀 Database integration for storing documents



