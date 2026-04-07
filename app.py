import streamlit as st
import spacy
from spacy import displacy
import pandas as pd
import re

# Page config
st.set_page_config(
    page_title="Entity Recognition",
    page_icon="🔍",
    layout="wide"
)

@st.cache_resource
def load_model(model_name):
    """Load spaCy model (cached)."""
    try:
        return spacy.load(model_name)
    except:
        st.warning(f"Model {model_name} not found. Trying en_core_web_sm...")
        try:
            return spacy.load("en_core_web_sm")
        except:
            st.error("No model found. Run: python -m spacy download en_core_web_sm")
            return None

def preprocess_text(text):
    """Clean and normalize text."""
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.replace('\n', ' ')
    return text

def calculate_confidence(ent):
    """Calculate confidence score for entity."""
    confidence = 0.7
    
    if ent.text[0].isupper():
        confidence += 0.1
    if len(ent.text.split()) > 1:
        confidence += 0.1
    if ent.label_ in ['DATE', 'MONEY', 'PERCENT', 'CARDINAL']:
        confidence += 0.1
    
    return min(confidence, 1.0)

def main():
    st.title("🔍 Named Entity Recognition")
    st.markdown("Extract entities from text using advanced NLP models")
    
    # Sidebar
    st.sidebar.header("Settings")
    
    # Model selection
    model_choice = st.sidebar.selectbox(
        "Select Model",
        options=["en_core_web_lg", "en_core_web_md", "en_core_web_sm"],
        help="Larger models = better accuracy but slower"
    )
    
    # Load model
    nlp = load_model(model_choice)
    if not nlp:
        return
    
    # Confidence threshold
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Filter entities below this confidence"
    )
    
    show_labels = st.sidebar.multiselect(
        "Filter Entity Types",
        options=["PERSON", "ORG", "GPE", "DATE", "MONEY", "TIME", "PRODUCT", "EVENT", "CARDINAL", "PERCENT"],
        default=["PERSON", "ORG", "GPE", "DATE", "MONEY"]
    )
    
    # Preprocessing option
    use_preprocessing = st.sidebar.checkbox("Enable Text Preprocessing", value=True)
    
    # Input
    text_input = st.text_area(
        "Enter text to analyze:",
        value="Apple Inc. was founded by Steve Jobs in Cupertino, California on April 1, 1976.",
        height=150
    )
    
    if st.button("Analyze", type="primary"):
        if text_input:
            doc = nlp(text_input)
            
            # Filter entities
            filtered_ents = [ent for ent in doc.ents if ent.label_ in show_labels]
            
            # Display visualization
            st.subheader("Visual Representation")
            if filtered_ents:
                doc.ents = filtered_ents
                html = displacy.render(doc, style="ent", jupyter=False)
                st.markdown(html, unsafe_allow_html=True)
            else:
                st.info("No entities found with selected filters")
            
            # Display table
            st.subheader("Extracted Entities")
            if filtered_ents:
                data = []
                for ent in filtered_ents:
                    data.append({
                        "Entity": ent.text,
                        "Type": ent.label_,
                        "Description": spacy.explain(ent.label_)
                    })
                st.table(data)
            
            # Stats
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Entities", len(filtered_ents))
            col2.metric("Unique Types", len(set(ent.label_ for ent in filtered_ents)))
            col3.metric("Words", len(doc))
        else:
            st.warning("Please enter some text")
    
    # Examples
    with st.expander("📝 Try Example Texts"):
        examples = [
            "Apple Inc. was founded by Steve Jobs in Cupertino, California on April 1, 1976.",
            "The meeting is scheduled for next Monday at 3 PM in New York.",
            "Elon Musk's company Tesla is valued at over $800 billion.",
            "Microsoft announced a partnership with OpenAI in January 2023."
        ]
        for i, example in enumerate(examples, 1):
            if st.button(f"Example {i}", key=f"ex{i}"):
                st.session_state.example = example
                st.rerun()

if __name__ == "__main__":
    main()
