import sys
import streamlit as st
from qna_utils import *
from io import BytesIO

st.set_page_config(page_title="DocuMind - Smart Document QnA System", layout="wide")

@st.cache_resource
def get_cached_models():
    return load_models()

sbert_model, summarizer = get_cached_models()

if "qna_history" not in st.session_state:
    st.session_state.qna_history = []

st.markdown("""
    <style>
    .main { background-color: #f3f4f6; }
    .stApp { padding: 2rem; }
    .card { background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); }
    .card h3 { color: #003566; }
    .stButton>button { width: 100%; }
    </style>
""", unsafe_allow_html=True)

st.title("📘 DocuMind: Smart Document QnA System")

# --- Sidebar
st.sidebar.header("📂 Upload Document")
input_method = st.sidebar.radio("Choose input type:", ["📄 Upload PDF", "✍️ Paste Text"])
text = ""
if input_method == "📄 Upload PDF":
    pdf = st.sidebar.file_uploader("Upload PDF", type="pdf")
    if pdf:
        text = load_pdf(pdf)
else:
    text = st.sidebar.text_area("Paste your document text:", height=200)

# --- Main Logic
if text:
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🔍 Preview")
        st.info(f"**Word Count:** {count_words(text)}")
        st.write(text[:500] + "...")
        st.markdown("</div>", unsafe_allow_html=True)

    @st.cache_data
    def get_embeddings(text):
        return precompute_embeddings(text, sbert_model)

    sentences, sentence_embeddings = get_embeddings(text)

    def extract_answer_from_context(context, question):
        return f"Based on the document, here is the most relevant information for your question:\n\n{context}"

    # --- Summary
    if st.button("📝 Generate Summary"):
        with st.spinner("Generating summary..."):
            summary = summarize_text(text, summarizer)
            st.session_state.summary = summary
            st.success("Summary Ready")
            with st.expander("📃 View Summary"):
                st.markdown(summary)
                st.download_button("⬇️ Download (.txt)", summary, file_name="summary.txt")

    # --- QnA Section
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("💬 Ask Questions")
    multi_q = st.toggle("Enable Multi-Question Mode")

    if not multi_q:
        question = st.text_input("Enter your question:")
        if question:
            with st.spinner("Searching for answer..."):
                relevant_text, score = find_relevant_sentences_from_embeddings(question, sentences, sentence_embeddings, sbert_model)
                answer = extract_answer_from_context(relevant_text, question)
                st.markdown(answer)
                st.info(f"Context relevance score: {score}%")
                st.download_button("Download Answer (.txt)", answer, file_name="answer.txt")
                st.session_state.qna_history.append({"q": question, "a": answer, "s": score})
    else:
        multi_input = st.text_area("Enter multiple questions (one per line):", height=150)
        if multi_input:
            questions = [q.strip() for q in multi_input.split("\n") if q.strip()]
            all_qa = []
            if st.button("🔎 Get Answers"):
                for q in questions:
                    relevant_text, score = find_relevant_sentences_from_embeddings(q, sentences, sentence_embeddings, sbert_model)
                    answer = extract_answer_from_context(relevant_text, q)
                    st.session_state.qna_history.append({"q": q, "a": answer, "s": score})
                    all_qa.append(f"Q: {q}\nA: {answer}\nConfidence: {score:.2f}%\n")
                for block in all_qa:
                    st.markdown(f"```\n{block}\n```")
                full_txt = "\n".join(all_qa)
                st.download_button("Download All (.txt)", full_txt, file_name="all_answers.txt")
    st.markdown("</div>", unsafe_allow_html=True)

    # --- History
    if st.session_state.qna_history:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📃 Question History")
        for i, qa in enumerate(st.session_state.qna_history[::-1]):
            with st.expander(f"Q{i+1}: {qa['q']}"):
                st.markdown(qa['a'])
                st.markdown(f"Context Score: {qa['s']}%")
        st.markdown("</div>", unsafe_allow_html=True)
