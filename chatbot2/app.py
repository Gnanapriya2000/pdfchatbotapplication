import streamlit as st
import requests

# FastAPI backend URL
BACKEND_URL = "http://127.0.0.1:8009"

st.set_page_config(page_title="PDF Chatbot", layout="centered")

st.title("📄 Chat with Your Document")

# Upload PDF
st.sidebar.header("📂 Upload PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    with st.sidebar:
        st.success("✅ PDF uploaded successfully!")
    
    files = {"file": uploaded_file.getvalue()}
    response = requests.post(f"{BACKEND_URL}/upload_pdf/", files=files)

    if response.status_code == 200:
        st.sidebar.success("✅ Document processed!")
    else:
        st.sidebar.error("❌ Error processing document.")

# Chatbot interface
st.subheader("💬 Ask a Question")
query = st.text_input("Type your question here...")

if st.button("Ask"):
    if not query:
        st.warning("⚠️ Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            response = requests.post(f"{BACKEND_URL}/query_pdf/", data={"query": query})
            if response.status_code == 200:
                answer = response.json().get("response", "❌ No response.")

    # Remove unnecessary parts
    if "**Question:**" in answer:
        answer = answer.split("**Question:**")[0]  # Remove past questions
    if "--- END OF ANSWER ---" in answer:
        answer = answer.replace("--- END OF ANSWER ---", "").strip()

    # Ensure it's properly formatted without extra spacing
    answer = answer.replace("\n\n", "\n").strip()  

    # Display answer properly
    st.markdown(f"🤖 **Answer:** {answer}")
