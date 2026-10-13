import streamlit as st
import pandas as pd

from backend.predictor import predict_cyberbullying
from backend.explainer import explain_text


st.set_page_config(
    page_title="Explainable AI",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Explainable AI")
st.write(
    "Understand why the mBERT model classified a comment as "
    "Cyberbullying or Non-Cyberbullying using LIME."
)

st.markdown("---")


# ============================================================
# MODE SELECTION
# ============================================================

mode = st.radio(
    "Choose input type:",
    ["Single Comment", "CSV File"],
    horizontal=True
)


# ============================================================
# SINGLE COMMENT
# ============================================================

if mode == "Single Comment":

    st.subheader("🔍 Analyze a Comment")

    text = st.text_area(
        "Enter a comment:",
        height=120,
        placeholder="Example: Go away idiot, nobody wants you here."
    )

    if st.button("Explain Prediction", type="primary"):

        if not text.strip():
            st.warning("Please enter a comment first.")

        else:

            with st.spinner("Analyzing comment..."):

                result = predict_cyberbullying(text)

            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            st.markdown("### Model Prediction")

            col1, col2 = st.columns(2)

            with col1:

                if result["label"] == "Cyberbullying":
                    st.error(f"🚨 {result['label']}")
                else:
                    st.success(f"✅ {result['label']}")

            with col2:

                st.metric(
                    "Confidence",
                    f"{result['confidence'] * 100:.2f}%"
                )

            # ------------------------------------------------
            # Probability Distribution
            # ------------------------------------------------

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

            # ------------------------------------------------
            # LIME
            # ------------------------------------------------

            st.markdown("---")
            st.markdown("### 🔬 LIME Explanation")

            with st.spinner("Generating explanation..."):

                explanation = explain_text(text)

            predicted_class = (
                1 if result["label"] == "Cyberbullying" else 0
            )

            available_labels = explanation.available_labels()

            if available_labels:
                explanation_list = explanation.as_list(label=available_labels[0])

                # Continue with your existing code for displaying
                # supporting and opposing words
            else:
                st.error("LIME could not generate an explanation for this comment.")
                explanation_list = []

            positive_words = [
                (word, weight)
                for word, weight in explanation_list
                if weight > 0
            ]

            negative_words = [
                (word, weight)
                for word, weight in explanation_list
                if weight < 0
            ]

            col1, col2 = st.columns(2)

            # ------------------------------------------------
            # Supporting Words
            # ------------------------------------------------

            with col1:

                st.subheader("🟢 Words Supporting the Prediction")

                if positive_words:

                    for word, weight in positive_words:

                        st.write(
                            f"**{word}** → +{weight:.4f}"
                        )

                        st.progress(
                            min(abs(weight), 1.0)
                        )

                else:

                    st.info(
                        "No strongly supporting words were identified."
                    )

            # ------------------------------------------------
            # Opposing Words
            # ------------------------------------------------

            with col2:

                st.subheader("🔵 Words Opposing the Prediction")

                if negative_words:

                    for word, weight in negative_words:

                        st.write(
                            f"**{word}** → {weight:.4f}"
                        )

                        st.progress(
                            min(abs(weight), 1.0)
                        )

                else:

                    st.info(
                        "No strongly opposing words were identified."
                    )

            st.info(
                "LIME explanations show how individual words contributed "
                "to this specific prediction. They should not be interpreted "
                "as universal cyberbullying keywords."
            )


# ============================================================
# CSV EXPLAINABILITY
# ============================================================

else:

    st.subheader("📂 Explain Comments from a CSV")

    st.write(
        "Upload a CSV file containing a column named **text**. "
        "The model will classify the comments and you can select "
        "an individual comment to generate its LIME explanation."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            df = pd.read_csv(uploaded_file)

            # ------------------------------------------------
            # Validate CSV
            # ------------------------------------------------

            if df.empty:

                st.error("The uploaded CSV file is empty.")
                st.stop()

            if "text" not in df.columns:

                st.error(
                    "The CSV file must contain a column named 'text'."
                )

                st.write(
                    "Columns found:",
                    list(df.columns)
                )

                st.stop()

            # Remove missing values
            df = df.dropna(subset=["text"]).copy()

            # Convert to string
            df["text"] = df["text"].astype(str)

            # Remove empty comments
            df = df[df["text"].str.strip() != ""]

            if df.empty:

                st.error(
                    "No valid comments were found in the 'text' column."
                )

                st.stop()

            # Reset index so row numbers are clean
            df = df.reset_index(drop=True)

            # ------------------------------------------------
            # Run Predictions
            # ------------------------------------------------

            st.markdown("### 📊 CSV Predictions")

            results = []

            progress_bar = st.progress(0)

            for i, text in enumerate(df["text"]):

                result = predict_cyberbullying(text)

                results.append({
                    "Prediction": result["label"],
                    "Confidence": result["confidence"]
                })

                progress_bar.progress(
                    (i + 1) / len(df)
                )

            progress_bar.empty()

            results_df = pd.DataFrame(results)

            display_df = df.copy()

            display_df["Prediction"] = results_df["Prediction"]

            display_df["Confidence"] = (
                results_df["Confidence"] * 100
            ).round(2)

            # ------------------------------------------------
            # Summary
            # ------------------------------------------------

            st.markdown("### 📈 Summary")

            total = len(display_df)

            cyberbullying_count = (
                display_df["Prediction"]
                .eq("Cyberbullying")
                .sum()
            )

            non_cyberbullying_count = (
                display_df["Prediction"]
                .eq("Non-Cyberbullying")
                .sum()
            )

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Comments", total)

            with col2:
                st.metric(
                    "Cyberbullying",
                    cyberbullying_count
                )

            with col3:
                st.metric(
                    "Non-Cyberbullying",
                    non_cyberbullying_count
                )

            with col4:

                rate = (
                    cyberbullying_count / total
                ) * 100

                st.metric(
                    "Cyberbullying Rate",
                    f"{rate:.2f}%"
                )

            # ------------------------------------------------
            # Results Table
            # ------------------------------------------------

            st.markdown("### 📋 Results")

            st.table(display_df)

            # ------------------------------------------------
            # Select Comment
            # ------------------------------------------------

            st.markdown("---")
            st.markdown("### 🔬 Select a Comment to Explain")

            selected_index = st.selectbox(
                "Choose a row:",
                options=list(range(len(display_df))),
                format_func=lambda i: (
                    f"Row {i + 1}: "
                    f"{display_df.loc[i, 'text'][:100]}"
                )
            )

            selected_text = display_df.loc[
                selected_index,
                "text"
            ]

            st.markdown("### Selected Comment")

            st.info(selected_text)

            # ------------------------------------------------
            # Selected Prediction
            # ------------------------------------------------

            selected_prediction = display_df.loc[
                selected_index,
                "Prediction"
            ]

            selected_confidence = display_df.loc[
                selected_index,
                "Confidence"
            ]

            col1, col2 = st.columns(2)

            with col1:

                if selected_prediction == "Cyberbullying":
                    st.error(
                        f"🚨 {selected_prediction}"
                    )
                else:
                    st.success(
                        f"✅ {selected_prediction}"
                    )

            with col2:

                st.metric(
                    "Confidence",
                    f"{selected_confidence:.2f}%"
                )

            # ------------------------------------------------
            # Generate Explanation
            # ------------------------------------------------

            if st.button(
                "🧠 Generate LIME Explanation",
                type="primary"
            ):

                with st.spinner(
                    "Generating LIME explanation..."
                ):

                    explanation = explain_text(
                        selected_text
                    )

                predicted_class = (
                    1
                    if selected_prediction == "Cyberbullying"
                    else 0
                )

                explanation_list = explanation.as_list(
                    label=predicted_class
                )

                positive_words = [
                    (word, weight)
                    for word, weight in explanation_list
                    if weight > 0
                ]

                negative_words = [
                    (word, weight)
                    for word, weight in explanation_list
                    if weight < 0
                ]

                # ------------------------------------------------
                # Explanation
                # ------------------------------------------------

                st.markdown("---")
                st.markdown("### 🧠 LIME Explanation")

                col1, col2 = st.columns(2)

                # Supporting
                with col1:

                    st.subheader(
                        "🟢 Words Supporting the Prediction"
                    )

                    if positive_words:

                        for word, weight in positive_words:

                            st.write(
                                f"**{word}** → +{weight:.4f}"
                            )

                            st.progress(
                                min(abs(weight), 1.0)
                            )

                    else:

                        st.info(
                            "No strongly supporting words "
                            "were identified."
                        )

                # Opposing
                with col2:

                    st.subheader(
                        "🔵 Words Opposing the Prediction"
                    )

                    if negative_words:

                        for word, weight in negative_words:

                            st.write(
                                f"**{word}** → {weight:.4f}"
                            )

                            st.progress(
                                min(abs(weight), 1.0)
                            )

                    else:

                        st.info(
                            "No strongly opposing words "
                            "were identified."
                        )

                st.info(
                    "LIME explanations describe the local contribution "
                    "of words for the selected comment. They should not "
                    "be interpreted as universal cyberbullying keywords."
                )

        except Exception as e:

            st.error(
                "Unable to process the uploaded CSV file."
            )
            st.exception(e)

            st.caption(str(e))