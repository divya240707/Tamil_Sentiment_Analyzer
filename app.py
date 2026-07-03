import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

st.set_page_config(page_title="Tamil Sentiment Analyzer", page_icon="🎬", layout="centered")

MODEL_PATH = "model/saved_model"

LABEL_COLORS = {
    "positive": "#1DB954",
    "negative": "#E63946",
    "neutral": "#A8A8A8",
}

LABEL_EMOJI = {
    "positive": "😊",
    "negative": "😞",
    "neutral": "😐",
}


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model


def predict(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
    label_id = torch.argmax(probs).item()
    label = model.config.id2label[label_id]
    confidence = probs[label_id].item()
    return label, confidence, probs


def main():
    st.title("🎬 Tamil Sentiment Analyzer")
    st.markdown(
        "Type any Tamil text — a movie review, comment, or sentence — "
        "and find out if it's **positive, negative, or neutral**."
    )

    tokenizer, model = load_model()

    example = st.selectbox(
        "Try an example, or type your own below:",
        [
            "",
            "படம் அருமையாக இருந்தது",
            "சேவை மிகவும் மோசமாக இருந்தது",
            "நாளைய கூட்டம் காலை பத்து மணிக்கு நடைபெறும்",
        ],
    )

    user_input = st.text_area("Enter Tamil text:", value=example, height=100)

    if st.button("Analyze Sentiment", type="primary"):
        if not user_input.strip():
            st.warning("Please enter some Tamil text first.")
        else:
            label, confidence, probs = predict(user_input, tokenizer, model)
            color = LABEL_COLORS.get(label, "#888")
            emoji = LABEL_EMOJI.get(label, "")

            st.markdown(
                f"<h2 style='color:{color}'>{emoji} {label.upper()}</h2>",
                unsafe_allow_html=True,
            )
            st.progress(confidence)
            st.caption(f"Confidence: {confidence*100:.1f}%")

            with st.expander("See full probability breakdown"):
                for idx, p in enumerate(probs):
                    lbl = model.config.id2label[idx]
                    st.write(f"{lbl}: {p.item()*100:.1f}%")

    st.markdown("---")
    st.caption("Built with MuRIL (Google) fine-tuned on Tamil sentiment data · Streamlit")


if __name__ == "__main__":
    main()
