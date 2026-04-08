import streamlit as st
import spacy
from spacy import displacy
import pandas as pd
import re

# Page config
st.set_page_config(
    page_title="Premium Entity Recognition",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for Premium Look
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stApp {
        background: radial-gradient(circle at top right, #1e293b, #0f172a);
    }
    .stSidebar {
        background-color: rgba(30, 41, 59, 0.7) !important;
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    h1 {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
    }
    .stButton>button {
        background: linear-gradient(90deg, #3b82f6, #2563eb);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 0.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }
    .css-1r6slb0 { /* Sidebar width */
        width: 350px;
    }
    .entity-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(model_name):
    """Load spaCy model (cached)."""
    try:
        return spacy.load(model_name)
    except:
        st.sidebar.error(f"⚠️ Model {model_name} could not be loaded.")
        # Try fallback order
        fallbacks = ["en_core_web_trf", "en_core_web_lg", "en_core_web_sm"]
        for fb in fallbacks:
            if fb != model_name:
                try:
                    m = spacy.load(fb)
                    st.sidebar.info(f"🔄 Fell back to {fb}")
                    return m
                except:
                    continue
        st.error("❌ No models found. Please download one using the command line.")
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
        "Select AI Model",
        options=["en_core_web_trf", "en_core_web_lg", "en_core_web_md", "en_core_web_sm"],
        index=0,
        help="Transformer (trf) is the most accurate but slowest. Large (lg) is a great balance."
    )
    
    st.sidebar.markdown("""
    ### Model Guide
    - **TRF**: Best for complex text and high accuracy.
    - **LG/MD**: Good for general purpose entity extraction.
    - **SM**: High speed, lower accuracy.
    """)
    
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
            # Preprocess if enabled
            processed_text = preprocess_text(text_input) if use_preprocessing else text_input
            
            with st.spinner("Analyzing text..."):
                doc = nlp(processed_text)
                
                # Calculate confidence and filter entities
                entities_with_conf = []
                for ent in doc.ents:
                    if ent.label_ in show_labels:
                        conf = calculate_confidence(ent)
                        if conf >= confidence_threshold:
                            entities_with_conf.append((ent, conf))
                
                filtered_ents = [ent for ent, _ in entities_with_conf]
                
                # Display visualization
                st.subheader("Visual Representation")
                if filtered_ents:
                    doc.ents = filtered_ents
                    html = displacy.render(doc, style="ent", jupyter=False)
                    st.markdown(html, unsafe_allow_html=True)
                else:
                    st.info("No entities found with selected filters")
                
                # Display table with confidence
                st.subheader("Extracted Entities")
                if entities_with_conf:
                    data = []
                    for ent, conf in entities_with_conf:
                        data.append({
                            "Entity": ent.text,
                            "Type": ent.label_,
                            "Confidence": f"{conf:.2%}",
                            "Position": f"{ent.start_char}-{ent.end_char}",
                            "Description": spacy.explain(ent.label_)
                        })
                    df = pd.DataFrame(data)
                    st.dataframe(df, width="stretch")
                    
                    # Download option
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name="entities.csv",
                        mime="text/csv"
                    )
                
                # Stats
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Entities", len(filtered_ents))
                col2.metric("Unique Types", len(set(ent.label_ for ent in filtered_ents)))
                col3.metric("Words", len(doc))
                col4.metric("Avg Confidence", f"{sum(c for _, c in entities_with_conf) / len(entities_with_conf):.2%}" if entities_with_conf else "N/A")
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
