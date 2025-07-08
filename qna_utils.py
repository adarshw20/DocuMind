import pdfplumber
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

nltk.download('punkt')

# Load SBERT and Summarization Model
def load_models():
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return sbert_model, summarizer

# Extract text from PDF
def load_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

# Summarize a large chunk of text
def summarize_text(text, summarizer, max_length=100, min_length=30):
    if len(text) > 3000:
        text = text[:3000]
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

# Compute sentence embedding using SBERT
def get_sentence_embedding(sentence, sbert_model):
    return sbert_model.encode(sentence, convert_to_numpy=True)

# Format answers into markdown
def format_answer_markdown(answer):
    answer = re.sub(r"(?<!\n)\s*(\d+[\.\)])\s+", r"\n\1 ", answer)
    answer = re.sub(r"(?<!\n)\s*[-*\u2022]\s+", r"\n• ", answer)
    return answer.strip()

# Precompute all embeddings for text
def precompute_embeddings(text, sbert_model):
    sentences = sent_tokenize(text)
    s_embeds = np.vstack([get_sentence_embedding(s, sbert_model) for s in sentences])
    return sentences, s_embeds

# Find top N relevant sentences with SBERT + cosine similarity
def find_relevant_sentences_from_embeddings(question, sentences, s_embeds, sbert_model, top_n=3, surrounding_sentences=2):
    q_embed = get_sentence_embedding(question, sbert_model).reshape(1, -1)
    similarities = cosine_similarity(s_embeds, q_embed)

    top_indices = similarities[:, 0].argsort()[-top_n:][::-1]
    confidence = similarities[top_indices[0], 0] * 100

    relevant = []
    for idx in top_indices:
        start = max(0, idx - surrounding_sentences)
        end = min(len(sentences), idx + surrounding_sentences + 1)
        relevant.extend(sentences[start:end])

    answer = " ".join(relevant)
    return format_answer_markdown(answer), round(confidence, 2)

# Count total words in the input text
def count_words(text):
    return len(text.split())
