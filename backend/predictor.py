import torch
import streamlit as st
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# --------------------------------------------------
# Model path
# --------------------------------------------------

MODEL_PATH = Path(__file__).resolve().parent.parent / "model"


# --------------------------------------------------
# Load model only once
# --------------------------------------------------

@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH)
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_PATH)
    )

    model.eval()

    return tokenizer, model


# Load tokenizer and model
tokenizer, model = load_model()


# --------------------------------------------------
# Labels
# --------------------------------------------------

LABELS = {
    0: "Non-Cyberbullying",
    1: "Cyberbullying"
}


# --------------------------------------------------
# Single-text prediction
# --------------------------------------------------

def predict_cyberbullying(text):
    """
    Predict whether a single text is cyberbullying.
    """

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():

        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )

        prediction = torch.argmax(
            probabilities,
            dim=-1
        ).item()

    confidence = probabilities[0][prediction].item()

    return {
        "label": LABELS[prediction],
        "confidence": confidence,
        "probabilities": {
            "Non-Cyberbullying": probabilities[0][0].item(),
            "Cyberbullying": probabilities[0][1].item()
        }
    }


# --------------------------------------------------
# Probability prediction for LIME
# --------------------------------------------------

def predict_proba(texts):
    """
    Return class probabilities for a list of texts.

    Used by LIME.

    Output:
        [
            [prob_non_cyberbullying, prob_cyberbullying],
            ...
        ]
    """

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():

        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )

    return probabilities.cpu().numpy()