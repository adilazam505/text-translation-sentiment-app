import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# -------------------------------
# Load Translation Model (lightweight)
# -------------------------------
@st.cache_resource
def load_translation_model():
    model_name = "Helsinki-NLP/opus-mt-en-ur"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

# -------------------------------
# Load Sentiment Model (lightweight)
# -------------------------------
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis")

# -------------------------------
# Translate Function
# -------------------------------
def translate_to_urdu(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=128)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

# -------------------------------
# Streamlit App
# -------------------------------
st.set_page_config(page_title="English → Urdu Translator + Sentiment", page_icon="🌍")
st.title("🌍 English → Urdu Translator + Sentiment Analysis")

# Load models
tokenizer, translation_model = load_translation_model()
sentiment_analyzer = load_sentiment_model()

# User Input
english_text = st.text_area("✍️ Enter an English sentence:", "")

if st.button("Translate & Analyze"):
    if english_text.strip():
        with st.spinner("Translating to Urdu..."):
            urdu_translation = translate_to_urdu(english_text, tokenizer, translation_model)

        with st.spinner("Analyzing Sentiment..."):
            sentiment = sentiment_analyzer(english_text)[0]

        # Results
        st.subheader("✅ Urdu Translation:")
        st.success(urdu_translation)

        st.subheader("🧐 Sentiment Analysis (English sentence):")
        st.info(f"Label: **{sentiment['label']}**, Score: **{sentiment['score']:.2f}**")
    else:
        st.warning("⚠️ Please enter a sentence first.")
