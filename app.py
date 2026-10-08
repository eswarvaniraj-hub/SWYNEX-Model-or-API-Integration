"""
AI Text Analyzer
Sentiment analysis powered by Hugging Face Transformers.
Simple, clean, modern UI for model integration demonstration.
"""

import streamlit as st
from transformers import pipeline

# -----------------------------------------------------------------------------
# 1. Page Configuration & Clean Light Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Text Analyzer",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom light, minimal CSS styling for a professional developer project
st.markdown("""
<style>
    /* Hide sidebar completely */
    [data-testid="stSidebar"] { display: none; }
    [data-testid="collapsedControl"] { display: none; }
    
    /* Global App Container */
    .block-container {
        max-width: 760px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }
    
    /* Header Typography */
    .app-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.25rem;
    }
    .app-subtitle {
        font-size: 1.1rem;
        font-weight: 500;
        color: #2563eb;
        margin-bottom: 0.5rem;
    }
    .app-description {
        font-size: 0.95rem;
        color: #64748b;
        margin-bottom: 1.75rem;
    }
    
    /* Card Containers */
    .content-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    
    /* Result Badges */
    .result-box-positive {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 10px;
        padding: 1.25rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .result-box-negative {
        background-color: #fef2f2;
        border: 1px solid #fecaca;
        border-radius: 10px;
        padding: 1.25rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .badge-label-positive {
        color: #15803d;
        font-size: 1.4rem;
        font-weight: 700;
    }
    .badge-label-negative {
        color: #b91c1c;
        font-size: 1.4rem;
        font-weight: 700;
    }
    .confidence-text {
        color: #334155;
        font-size: 1.05rem;
        font-weight: 600;
        margin-top: 0.35rem;
    }

    /* Model Info & Footer */
    .model-info-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-top: 1.5rem;
        font-size: 0.9rem;
        color: #475569;
    }
    .app-footer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.85rem;
        margin-top: 2.5rem;
        padding-top: 1.25rem;
        border-top: 1px solid #f1f5f9;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. Model Loading (Cached with Streamlit)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading sentiment analysis model...")
def load_sentiment_model():
    """Load and cache the Hugging Face sentiment analysis pipeline."""
    try:
        return pipeline(
            "sentiment-analysis",
            model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
        )
    except Exception as e:
        return None


# -----------------------------------------------------------------------------
# 3. Header Section
# -----------------------------------------------------------------------------
st.markdown('<div class="app-title">🤖 AI Text Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Sentiment analysis powered by Hugging Face Transformers</div>', unsafe_allow_html=True)
st.markdown('<div class="app-description">Enter text below and let the AI model determine its sentiment.</div>', unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 4. State Initialization
# -----------------------------------------------------------------------------
if "text_input" not in st.session_state:
    st.session_state["text_input"] = "I really enjoyed using this application!"


# -----------------------------------------------------------------------------
# 5. Main Analyzer Card
# -----------------------------------------------------------------------------
st.markdown("### Analyze Text")

user_text = st.text_area(
    label="Enter your text",
    value=st.session_state["text_input"],
    placeholder="Example: I really enjoyed using this application!",
    height=130,
    key="text_area_input"
)

# Keep session state in sync
st.session_state["text_input"] = user_text

analyze_btn = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)


# -----------------------------------------------------------------------------
# 6. Result Section (Only shown after user clicks Analyze Sentiment)
# -----------------------------------------------------------------------------
if analyze_btn:
    if not user_text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        classifier = load_sentiment_model()
        if classifier is None:
            st.error("The Hugging Face model could not be loaded. Please ensure required dependencies are installed.")
        else:
            with st.spinner("Analyzing text..."):
                raw_output = classifier(user_text[:512])
                result = raw_output[0]
                label = result.get("label", "").upper()
                score = float(result.get("score", 0.0))
                confidence_pct = round(score * 100, 2)
            
            st.markdown("---")
            st.markdown("### Analysis Result")
            
            if "POSITIVE" in label:
                st.markdown(f"""
                <div class="result-box-positive">
                    <div class="badge-label-positive">😊 POSITIVE</div>
                    <div class="confidence-text">Confidence: {confidence_pct:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-box-negative">
                    <div class="badge-label-negative">☹️ NEGATIVE</div>
                    <div class="confidence-text">Confidence: {confidence_pct:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Optional expander for technical inspection
            with st.expander("View technical model output"):
                st.json(raw_output)


# -----------------------------------------------------------------------------
# 7. Sample Inputs Section
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("### Try Sample Inputs")
st.caption("Click a sample below to load it into the analyzer:")

sample_col1, sample_col2 = st.columns(2)

with sample_col1:
    st.markdown("**Positive Sample**")
    st.caption("Expected: POSITIVE")
    if st.button("Load Positive Sample", use_container_width=True):
        st.session_state["text_input"] = "I really enjoyed using this application!"
        st.rerun()

with sample_col2:
    st.markdown("**Negative Sample**")
    st.caption("Expected: NEGATIVE")
    if st.button("Load Negative Sample", use_container_width=True):
        st.session_state["text_input"] = "The application was slow and frustrating."
        st.rerun()


# -----------------------------------------------------------------------------
# 8. Model Information Section
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("### About the Model")
st.markdown("""
<div class="model-info-box">
    This application uses a pretrained Hugging Face Transformers sentiment-analysis model to classify text as positive or negative.<br/><br/>
    <b>Model:</b> <code>distilbert/distilbert-base-uncased-finetuned-sst-2-english</code>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 9. Footer
# -----------------------------------------------------------------------------
st.markdown('<div class="app-footer">Model or API Integration • Python • Streamlit • Hugging Face</div>', unsafe_allow_html=True)
