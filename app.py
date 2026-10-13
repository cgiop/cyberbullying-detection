import streamlit as st
from backend.predictor import predict_cyberbullying

st.set_page_config(
    page_title="Cyberbullying Detection System",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Cyberbullying Detection System")

st.markdown("""
### Project Description

This application detects whether a given text contains **cyberbullying content**
using a fine-tuned multilingual BERT (mBERT) model.

The system supports individual comment analysis, batch detection from CSV files,
and explainable AI using LIME to help understand the model's predictions.
""")

st.markdown("---")

st.markdown("### 🤖 Model")

st.markdown("""
The detection system uses **mBERT (`bert-base-multilingual-cased`)**
fine-tuned for binary text classification.

The model predicts two classes:

- **Cyberbullying**
- **Non-Cyberbullying**

Along with the predicted class, the application displays the model's
confidence and class probability distribution.
""")

st.markdown("---")

st.markdown("### ⚙️ Features")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🔍 Text Detection")
    st.write(
        "Analyze individual comments and classify them as "
        "Cyberbullying or Non-Cyberbullying."
    )

with col2:
    st.subheader("📊 Batch Detection")
    st.write(
        "Upload a CSV file containing multiple comments and analyze "
        "all comments together with summary statistics."
    )

with col3:
    st.subheader("🧠 Explainable AI")
    st.write(
        "Use LIME to understand which words contributed toward or "
        "against the model's prediction."
    )

st.markdown("---")

st.subheader("🔎 Analyze a Comment")

text = st.text_area(
    "Enter a comment:",
    height=120,
    placeholder="Type a comment here..."
)

if st.button("Analyze", type="primary"):

    if not text.strip():
        st.warning("Please enter a comment before analyzing.")

    else:
        with st.spinner("Analyzing comment..."):
            result = predict_cyberbullying(text)

        st.markdown("### Prediction")

        if result["label"] == "Cyberbullying":
            st.error(f"🚨 {result['label']}")
        else:
            st.success(f"✅ {result['label']}")

        st.metric(
            "Confidence",
            f"{result['confidence'] * 100:.2f}%"
        )

        st.markdown("### Probability Distribution")

        non_cb = result["probabilities"]["Non-Cyberbullying"]
        cb = result["probabilities"]["Cyberbullying"]

        col1, col2 = st.columns(2)

        with col1:
            st.write("Non-Cyberbullying")
            st.progress(non_cb)
            st.caption(f"{non_cb * 100:.2f}%")

        with col2:
            st.write("Cyberbullying")
            st.progress(cb)
            st.caption(f"{cb * 100:.2f}%")