# 🧠 Email Search AI — RAG-Based Semantic Email Query System

## 📋 Overview
**Email Search AI** is a Retrieval-Augmented Generation (RAG) based system that allows you to **search, understand, and summarize email data** using natural language queries.  
Instead of keyword-based search, it uses **semantic embeddings**, **vector similarity**, and **LLM-based generation** to provide context-aware results.

---

## ⚙️ Features
✅ Load emails directly from a `.csv` file  
✅ Store and search email content using **ChromaDB**  
✅ Use **Sentence Transformers** for embedding generation  
✅ Retrieve top semantic matches for a query  
✅ Use **OpenAI GPT** (or any compatible LLM) to summarize or answer questions based on retrieved emails  
✅ End-to-end runnable on **Google Colab** or **Jupyter Notebook**

---

## 🧩 Architecture

```
+-----------------------------+
|        CSV Loader           |
| (email dataset ingestion)   |
+-------------+---------------+
              |
              v
+-----------------------------+
|   SentenceTransformer       |
| (embedding generator)       |
+-------------+---------------+
              |
              v
+-----------------------------+
|        ChromaDB             |
| (vector storage + retrieval)|
+-------------+---------------+
              |
              v
+-----------------------------+
|   OpenAI GPT (Generation)   |
| (final natural-language ans)|
+-----------------------------+
```

---

## 🗂️ Folder Structure
```
📁 Email_Search_AI/
 ├── Email_Search_AI_Minimal_Clean.ipynb   # Main notebook
 ├── emails.csv                            # Sample email dataset (user-provided)
 ├── chroma_db/                            # Vector database storage
 └── README.md                             # Project documentation
```

---

## 🧰 Prerequisites

### 1. Install Dependencies
Run this cell in your notebook:
```bash
!pip install chromadb sentence-transformers openai pandas
```

### 2. Environment Variable
Set your OpenAI API Key before running:
```python
import os
os.environ["OPENAI_API_KEY"] = "sk-xxxxxx"  # replace with your key
```

---

## 🧮 How It Works

### Step 1 — Load Data
```python
import pandas as pd

df = pd.read_csv("emails.csv")
sample_texts = df['email'].dropna().tolist()
```

### Step 2 — Embed & Store in ChromaDB
```python
from sentence_transformers import SentenceTransformer
import chromadb

embedder = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("emails")

embeddings = embedder.encode(sample_texts).tolist()
collection.add(documents=sample_texts, embeddings=embeddings,
               ids=[f"email_{i}" for i in range(len(sample_texts))])
```

### Step 3 — Query & Retrieve
```python
query = "Who was responsible for the project delay?"
qvec = embedder.encode([query])[0].tolist()

results = collection.query(query_embeddings=[qvec], n_results=3)
top_docs = results['documents'][0]
```

### Step 4 — Generate Final Answer
```python
from openai import OpenAI
client = OpenAI()

context = "\n".join(top_docs)
prompt = f"Based on the following emails, answer: {query}\n\nEmails:\n{context}"

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)
print("🧩 Final Answer:\n", response.choices[0].message.content)
```

---

## 📊 Expected Output Example

### Query:
> “Who was responsible for the project delay?”

### Top 3 Retrieved Emails:
1. *“QA reported three critical bugs in the Kubernetes service.”*  
2. *“Project Pegasus was delayed due to resource constraints in Q3.”*  
3. *“The project lead confirmed the delay was due to dependency failures.”*

### Generated Answer:
> “The project delay was primarily caused by resource constraints and unresolved QA issues in the Kubernetes service.”

---

## 📸 Deliverables (for Project Report)
You can take **6 screenshots**:
1️⃣ Three showing **search layer (retrieval results)**  
2️⃣ Three showing **generation layer (final answers)**  

---

## 🚀 Future Enhancements
- Support for multiple data sources (emails, PDFs, meeting notes)  
- Multi-turn conversational retrieval  
- Integration with LangChain or LlamaIndex  
- Fine-tuned domain-specific embeddings  

---

## 👩‍💻 Author
**Deepika Ramesh**  
Email Search AI — 2025  
Designed for intelligent, context-aware enterprise search.
