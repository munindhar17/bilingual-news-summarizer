import streamlit as st
# Set page config must be the first Streamlit command
st.set_page_config(
    page_title="News Article Summarizer",
    page_icon="📰",
    layout="wide"
)

import nltk
import numpy as np
import networkx as nx
import json
import os
from indicnlp.tokenize import sentence_tokenize, indic_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from googletrans import Translator

# Download necessary NLTK resources (cached for performance)
@st.cache_resource
def download_nltk_resources():
    nltk.download('punkt')
    nltk.download('stopwords')
download_nltk_resources()

# Telugu stopwords (basic sample list; extend as needed)
telugu_stopwords = set(["తతర", "ఇ", "అ", "ఈ", "ఆ"])

# File to store history
history_file = "news_history.json"

# Load history from file
def load_history():
    try:
        if os.path.exists(history_file):
            with open(history_file, "r", encoding="utf-8") as file:
                return json.load(file)
        return []
    except Exception as e:
        st.error(f"Error loading history: {e}")
        return []

# Save history to file
def save_history(history):
    try:
        with open(history_file, "w", encoding="utf-8") as file:
            json.dump(history, file, ensure_ascii=False, indent=4)
    except Exception as e:
        st.error(f"Error saving history: {e}")

# Preprocess text for summarization
def preprocess_text(text, lang="en"):
    if not text:
        return [], []

    if lang == "te":  # Telugu processing
        try:
            sentences = sentence_tokenize.sentence_split(text, lang='te')
            if not sentences:
                sentences = text.split('.')
            sentences = [s.strip() for s in sentences if s.strip()]
            clean_sentences = [
                " ".join(
                    [word for word in indic_tokenize.trivial_tokenize(sent, lang='te')
                     if word not in telugu_stopwords]
                )
                for sent in sentences
            ]
        except Exception:
            sentences = text.split('.')
            sentences = [s.strip() for s in sentences if s.strip()]
            clean_sentences = sentences
    else:  # English or other languages
        try:
            sentences = nltk.sent_tokenize(text)
            stop_words = set(nltk.corpus.stopwords.words("english"))
            clean_sentences = [
                " ".join(
                    [word.lower() for word in nltk.word_tokenize(sent)
                     if word.isalnum() and word.lower() not in stop_words]
                )
                for sent in sentences
            ]
        except Exception:
            sentences = text.split('.')
            sentences = [s.strip() for s in sentences if s.strip()]
            clean_sentences = sentences

    return sentences, clean_sentences

# Build similarity matrix for extractive summarization
def build_similarity_matrix(sentences):
    if not sentences or all(not s for s in sentences):
        return np.zeros((1, 1))
    try:
        vectorizer = TfidfVectorizer()
        sentence_vectors = vectorizer.fit_transform(sentences)
        similarity_matrix = (sentence_vectors * sentence_vectors.T).toarray()
        return similarity_matrix
    except Exception:
        return np.zeros((len(sentences), len(sentences)))

# Extractive summarization using PageRank
def extractive_summarization(text, num_sentences=3, lang="en"):
    if not text:
        return "No text provided for summarization."

    original_sentences, clean_sentences = preprocess_text(text, lang)
    if not clean_sentences or len(clean_sentences) < 1:
        return "Error: Not enough valid sentences to generate a summary."

    if len(clean_sentences) <= num_sentences:
        return " ".join(original_sentences)

    try:
        similarity_matrix = build_similarity_matrix(clean_sentences)
        nx_graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(nx_graph)
        ranked_sentences = sorted(
            ((scores[i], sent) for i, sent in enumerate(original_sentences)),
            reverse=True
        )
        summary = " ".join(
            [ranked_sentences[i][1] for i in range(min(num_sentences, len(ranked_sentences)))]
        )
        return summary
    except Exception:
        return " ".join(original_sentences[:num_sentences])

# Basic language detection (English vs Telugu)
def detect_language(text):
    telugu_range = range(0x0C00, 0x0C7F)
    for char in text:
        if ord(char) in telugu_range:
            return "te"
    return "en"

# Translation wrapper
def translate_text(text, src_lang, dest_lang):
    if not text:
        return ""
    if src_lang == dest_lang:
        return text
    try:
        with st.spinner('Translating text...'):
            translator = Translator()
            translated = translator.translate(text, src=src_lang, dest=dest_lang)
            return translated.text
    except Exception:
        return text + " (Translation failed)"

# ---------------------- Streamlit UI ----------------------
st.title("📰 News Article Summarizer")
st.markdown("Summarize news articles in English and Telugu")

# Load history from file
history = load_history()

# Sidebar
st.sidebar.header("Options")
show_advanced = st.sidebar.checkbox("Show Advanced Options", False)
if show_advanced:
    st.sidebar.subheader("Advanced Settings")
    auto_detect = st.sidebar.checkbox("Auto-detect input language", True)
else:
    auto_detect = True

# Tabs for main app
tab1, tab2 = st.tabs(["Summarize Article", "View History"])

with tab1:
    text = st.text_area("Enter the news article for summarization:", height=250)

    col1, col2, col3 = st.columns(3)
    with col1:
        detected_lang = detect_language(text) if text else "en"
        if show_advanced and auto_detect and text:
            st.info(f"Detected language: {'Telugu' if detected_lang == 'te' else 'English'}")
            input_lang = detected_lang
        else:
            input_lang = st.selectbox(
                "Input Language", ["en", "te"],
                format_func=lambda x: "English" if x == "en" else "Telugu"
            )

    with col2:
        output_lang = st.selectbox(
            "Output Language", ["en", "te"],
            format_func=lambda x: "English" if x == "en" else "Telugu"
        )

    with col3:
        num_sentences = st.slider("Number of sentences in summary", 1, 10, 3)

    if st.button("Generate Summary"):
        if text:
            with st.spinner('Processing...'):
                summary = extractive_summarization(
                    text, num_sentences=num_sentences, lang=input_lang
                )
                # Translate if needed
                if input_lang != output_lang:
                    with st.spinner(f"Translating summary..."):
                        summary = translate_text(summary, input_lang, output_lang)

            # Display
            st.subheader("Generated Summary:")
            st.write(summary)

            # Save history
            history.append({
                "text": text,
                "summary": summary,
                "input_lang": input_lang,
                "output_lang": output_lang,
                "num_sentences": num_sentences
            })
            save_history(history)
            st.success("Summary generated and saved to history!")
        else:
            st.warning("Please enter text to summarize")

with tab2:
    st.subheader("Previously Summarized Articles")
    if not history:
        st.info("No history found. Summarize some articles to see them here.")
    else:
        for i, entry in enumerate(reversed(history)):
            with st.expander(f"Article {len(history) - i}"):
                st.markdown("**Original Text:**")
                full_text = entry.get("text", "")
                if len(full_text) > 500:
                    st.write(full_text[:500] + "...")
                else:
                    st.write(full_text)

                st.markdown("**Summary:**")
                st.write(entry.get("summary", "No summary available"))

                st.markdown("**Details:**")
                input_lang_full = "English" if entry.get("input_lang", "en") == "en" else "Telugu"
                output_lang_full = "English" if entry.get("output_lang", "en") == "en" else "Telugu"

                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"Input Language: {input_lang_full}")
                    st.write(f"Output Language: {output_lang_full}")
                with col2:
                    st.write(f"Number of sentences: {entry.get('num_sentences', 3)}")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear History"):
            if st.checkbox("I understand this will delete all history", key="confirm_delete", value=False):
                save_history([])
                st.success("History cleared!")
                st.experimental_rerun()
            else:
                st.warning("Please confirm deletion by checking the box")

    with col2:
        if st.button("Reset History File"):
            with open(history_file, "w", encoding="utf-8") as file:
                file.write("[]")
            st.success("History file has been reset to an empty array")
            st.experimental_rerun()
