📰 Bilingual News Article Summarizer

A **Streamlit** web application that summarizes news articles in **English** and **Telugu**, with optional translation using Google Translate.

🚀 Features
- Supports English & Telugu news
- Auto language detection
- Extractive summarization using **TF-IDF + PageRank**
- Translation between English and Telugu
- Adjustable summary length
- Saves and displays summary history

🖥 Run Locally
pip install streamlit nltk numpy networkx indic-nlp-library scikit-learn googletrans==4.0.0-rc1
streamlit run Text_news_summarizer.py
Then open the link shown in the terminal (usually http://localhost:8501).

📂 Files
Text_news_summarizer.py — Main application code
README.md — Documentation