# 📘 DocuMind: Smart Document Summarizer & QnA System

DocuMind is an intelligent Streamlit-based application that allows you to:

* 📄 Upload or paste large documents
* 📝 Summarize the content
* 💬 Ask questions and get context-aware answers from the document
* 📥 Download answers and summaries
* 🕘 Keep a session history of your questions and answers

This version is fully offline—no OpenAI or API dependencies—using locally hosted NLP models.

---

## 🚀 Features

* **PDF/Text Upload**: Load documents via upload or direct paste
* **Text Summarization**: Generate brief summaries using `facebook/bart-large-cnn`
* **Question Answering**: Get answers using semantic similarity via `sentence-transformers`
* **Multi-Question Mode**: Ask multiple questions at once
* **Answer Confidence**: Shows context relevance confidence score
* **Download Options**: Export answers/summaries to `.txt`
* **Fully Offline**: No internet or OpenAI API required

---

## 🧠 Tech Stack

| Component           | Library                                     |
| ------------------- | ------------------------------------------- |
| UI                  | Streamlit                                   |
| Summarization       | HuggingFace Transformers (`bart-large-cnn`) |
| Sentence Embeddings | Sentence-Transformers (`all-MiniLM-L6-v2`)  |
| PDF Parsing         | pdfplumber                                  |
| NLP                 | nltk, scikit-learn                          |

---

## 📦 Installation

1. **Clone the repo**

```bash
git clone https://github.com/yourusername/DocuMind.git
cd DocuMind
```

2. **Create a virtual environment** (optional but recommended)

```bash
python -m venv venv
venv\Scripts\activate  # On Windows
source venv/bin/activate  # On Linux/macOS
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Run the app**

```bash
streamlit run app.py
```

---

## 📁 File Structure

```
DocuMind/
├── app.py                 # Main Streamlit app
├── qna_utils.py           # Utility functions for QnA and summarization
├── requirements.txt       # Python dependencies
└── README.md              # You're here
```

---

## 📝 Example Use

1. Upload your research paper or thesis as a PDF
2. Click **Generate Summary** to get a quick overview
3. Ask specific questions about the content
4. Export answers to `.txt` for notes or reports

---

## 🔧 Future Improvements

* Export to PDF
* Optional GPT-based answering (with API key)
* Audio summary/answers
* User authentication

---

## 🙌 Credits

Built by [Adarsh Wankhede](https://github.com/adarshwankhede) with ❤️ for efficient academic research.

---

