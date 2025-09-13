📰 Bilingual Text News Summarizer: English & Telugu

A web-based application that summarizes lengthy news articles in English and Telugu, helping readers grasp key information quickly. The system supports bilingual input/output, customizable summary lengths, and real-time interaction via a Streamlit interface.

🚀 Features

Multilingual Input: Accepts both English and Telugu news articles.

Bilingual Summaries: Generates summaries in English, Telugu, or both.

Customizable Length: Slider to set summary length (1–10 sentences).

Real-Time Summarization: Instant results via Streamlit web app.

Translation Support: Integrated with Google Translate API for cross-language summaries.

History View: Stores previously generated summaries for reference.

🛠️ Tech Stack

Language: Python 3.10+

Frameworks & Libraries:

Streamlit
 – Interactive web app

NLTK
 – Tokenization, stopword removal

scikit-learn
 – TF-IDF vectorization

NetworkX
 – PageRank for extractive summarization

Indic-NLP Library
 – Telugu NLP tasks

Google Translate API
 – On-the-fly translations

💻 Requirements
Software

OS: Windows 10/11, macOS, or Linux (Ubuntu 20.04+)

Python 3.10 or above

Modern web browser (Chrome, Firefox, Edge)

Virtual environment recommended (venv or conda)

Hardware

CPU: Dual-core (i3/Ryzen 3) minimum, Quad-core (i5/i7) recommended

RAM: 4 GB minimum, 8 GB recommended

Storage: 1 GB free space

Display: 720p minimum (1080p recommended)

⚙️ Installation
# Clone the repository
git clone https://github.com/your-username/bilingual-news-summarizer.git
cd bilingual-news-summarizer

# Create virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt


Download required NLTK datasets:

import nltk
nltk.download('punkt')
nltk.download('stopwords')

▶️ Usage

Run the Streamlit app:

streamlit run app.py


Paste or type a news article into the text area.

Select input language (English/Telugu) or enable auto-detect.

Choose output language (English, Telugu, or both).

Adjust summary length using the slider.

Click Generate Summary → Get instant results!

📊 Example

Input (English):
A lengthy article about water disputes under the Indus Treaty.

Output (Telugu, 2 sentences):
A concise translation preserving the meaning of the original text.

🔮 Future Enhancements

Support for more Indian languages (Hindi, Tamil, Kannada).

Transformer-based abstractive summarization (BART, mBART fine-tuned).

Domain-specific summaries (finance, healthcare, sports).

Real-time cloud deployment (Streamlit Cloud / Heroku).

Content customization (topic-specific summarization).
