# 🎬 Tamil Sentiment Analyzer

Detects the sentiment — **positive, negative, or neutral** — of Tamil text using a fine-tuned multilingual transformer (Google's MuRIL).

🔗 **Live demo:** _add your Streamlit Cloud link here after deployment_

---

## Why this project?

Most sentiment analysis tools and tutorials are built for English. Tamil — spoken by over 75 million people — is underrepresented in NLP tooling despite a massive amount of digital content (Tamil cinema reviews, social media, news). This project fine-tunes a multilingual transformer specifically for Tamil sentiment classification, with movie/product reviews as the primary use case.

## Demo

> Type: `படம் அருமையாக இருந்தது` → **Positive 😊**
> Type: `சேவை மிகவும் மோசமாக இருந்தது` → **Negative 😞**

(Add a screenshot or GIF of the app here)

## How it works

1. Text is tokenized using MuRIL's multilingual tokenizer (trained on 17 Indian languages including Tamil)
2. The fine-tuned classification head predicts one of three sentiment classes
3. Results are shown with a confidence score and full probability breakdown

## Tech Stack

- **Model:** [google/muril-base-cased](https://huggingface.co/google/muril-base-cased) (HuggingFace Transformers)
- **Training:** PyTorch + HuggingFace Trainer
- **Frontend:** Streamlit
- **Deployment:** Streamlit Community Cloud

## Project Structure

```
tamil-sentiment-analyzer/
├── data/
│   └── sample_tamil_sentiment.csv   # labeled training data
├── model/
│   ├── train.py                     # fine-tuning script
│   └── saved_model/                 # output after training (not committed — see .gitignore)
├── app/
│   └── app.py                       # Streamlit application
├── requirements.txt
└── README.md
```

## Running locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/tamil-sentiment-analyzer.git
cd tamil-sentiment-analyzer

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (takes ~10-20 min on CPU for the sample dataset)
python model/train.py

# 4. Launch the app
streamlit run app/app.py
```

## Dataset

The sample dataset (`data/sample_tamil_sentiment.csv`) contains hand-labeled Tamil sentences across positive, negative, and neutral classes. For production use, this should be expanded with a larger labeled corpus (e.g. Tamil movie reviews, Twitter/X data).

## Limitations & Future Work

- Current dataset is small (~20 samples) — accuracy improves significantly with more labeled data
- Code-mixed Tamil-English text (common in real social media) is not yet well handled
- Future: expand dataset, add a confusion matrix evaluation, support for Tanglish (Tamil written in English script)

## Author

Divya — B.E. Computer Science, Sri Krishna College of Engineering and Technology, Coimbatore

---

*Built as part of a college mini-project, with a focus on regional language NLP.*
