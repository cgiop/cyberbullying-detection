
import numpy as np
import shap
import streamlit as st

from backend.predictor import predict_proba, tokenizer

CLASS_NAMES = [
    "Non-Cyberbullying",
    "Cyberbullying",
]


@st.cache_resource
def load_shap_explainer():
    masker = shap.maskers.Text(tokenizer)

    return shap.Explainer(
        predict_proba,
        masker,
        output_names=CLASS_NAMES,
    )


def explain_with_shap(text, max_evals=200):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Please provide a non-empty comment.")

    explainer = load_shap_explainer()

    probabilities = predict_proba([text])[0]
    predicted_class = int(np.argmax(probabilities))

    shap_values = explainer(
        [text],
        max_evals=max_evals,
    )

    explanation = shap_values[0, :, predicted_class]

    return explanation, predicted_class, probabilities