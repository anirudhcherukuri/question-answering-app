import streamlit as st
from transformers import pipeline

# Set page config
st.set_page_config(page_title="Question Answering App", page_icon="🤖", layout="wide")

# Custom background image CSS
st.markdown(
    """
    <style>
    .stApp {
        background: url(https://images.app.goo.gl/1L5D18i8wY8Z9FU48) no-repeat center center fixed;
        background-size: cover;
    }
    .block-container {
        background-color: rgba(255, 255, 255, 0.8);
        padding: 2rem;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Page title and description
st.title("🤖 Question Answering App")
st.write("Ask questions based on the provided context paragraph using a pre-trained NLP model (DistilBERT SQuAD).")

# Loading spinner while initializing model
with st.spinner("Loading QA model..."):
    qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Layout: Two columns for better organisation
col1, col2 = st.columns(2)

with col1:
    context = st.text_area("📝 Enter Context Paragraph", height=250, placeholder="Paste or type your context here...")

with col2:
    question = st.text_input("❓ Enter Your Question", placeholder="Type your question here...")

# Predict answer with styled output
if st.button("🔍 Get Answer"):
    if context and question:
        with st.spinner("Finding the answer..."):
            result = qa_pipeline(question=question, context=context)
            st.success(f"**Answer:** {result['answer']}")
    else:
        st.warning("⚠️ Please enter both context and question to get an answer.")

# Footer
st.markdown("---")
st.write("Made with ❤️ using Streamlit and Hugging Face Transformers")
