# Entity Recognition using NLP

A simple Named Entity Recognition (NER) project using spaCy.

## Setup

```bash
# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy models (choose based on your needs)
python -m spacy download en_core_web_sm   # Small, fast
python -m spacy download en_core_web_md   # Medium, balanced
python -m spacy download en_core_web_lg   # Large, most accurate
```

## Usage

### Web UI (Streamlit)
```bash
streamlit run app.py
```

### Command Line
```bash
python entity_recognition.py
```

## Features

- Recognizes entities: PERSON, ORG, GPE, DATE, MONEY, etc.
- Processes text and highlights entities
- Easy to extend with custom entity types
