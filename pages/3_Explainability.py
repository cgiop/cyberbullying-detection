import streamlit as st
import pandas as pd
import numpy as np
from backend.predictor import predict_cyberbullying
from backend.explainer import explain_text
from backend.shap_explainer import (
    explain_with_shap,
    CLASS_NAMES,
)
st.set_page_config(
    page_title="Explainable AI",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Explainable AI")
st.write(
    "Understand why the mBERT model classified a comment as "
    "Cyberbullying or Non-Cyberbullying."
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
explanation_method = st.radio(
    "Choose explanation method:",
    ["LIME", "SHAP"],
    horizontal=True,
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


            st.markdown("---")

            if explanation_method == "LIME":

                st.markdown("### 🔬 LIME Explanation")

                with st.spinner("Generating LIME explanation..."):
                    explanation = explain_text(text)

                available_labels = explanation.available_labels()

                if available_labels:
                    explanation_list = explanation.as_list(
                        label=available_labels[0]
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

                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("🟢 Supporting Words")

                        for word, weight in positive_words:
                            st.write(f"**{word}** → +{weight:.4f}")
                            st.progress(min(abs(weight), 1.0))

                    with col2:
                        st.subheader("🔵 Opposing Words")

                        for word, weight in negative_words:
                            st.write(f"**{word}** → {weight:.4f}")
                            st.progress(min(abs(weight), 1.0))

                else:
                    st.warning("LIME could not generate an explanation.")

            else:

                st.markdown("### 🧠 SHAP Explanation")

                with st.spinner("Generating SHAP explanation..."):
                    shap_exp, predicted_class, probabilities = (
                        explain_with_shap(text)
                    )

                st.write(
                    f"Explaining prediction: **{CLASS_NAMES[predicted_class]}**"
                )

                # Extract tokens and their SHAP values
                tokens = list(shap_exp.data)
                values = np.asarray(shap_exp.values).flatten()

                shap_df = pd.DataFrame({
                    "Token": tokens,
                    "SHAP Value": values,
                })

                shap_df = shap_df[
                    shap_df["Token"].astype(str).str.strip() != ""
                    ]

                shap_df = shap_df.sort_values(
                    "SHAP Value",
                    key=lambda x: x.abs(),
                    ascending=False,
                )

                st.markdown("#### Token Contributions")

                st.caption(
                    "Positive values push the selected class score higher; "
                    "negative values push it lower."
                )

                st.dataframe(
                    shap_df.reset_index(drop=True),
                    use_container_width=True,
                    hide_index=True,
                )

                st.markdown("#### Most Important Tokens")

                chart_df = shap_df.head(15).sort_values("SHAP Value")

                if not chart_df.empty:
                    st.bar_chart(
                        chart_df.set_index("Token")["SHAP Value"]
                    )

                st.info(
                    "SHAP values describe how tokens contribute to this "
                    "individual model prediction. They do not establish "
                    "that a word is universally associated with cyberbullying."
                )


# ============================================================
# CSV EXPLAINABILITY
# ============================================================

else:

    st.subheader("📂 Explain Comments from a CSV")

    st.write(
        "Upload a CSV file containing a column named **text**. "
        "The model will classify the comments and you can select "
        "an individual comment to generate its explanation."
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
                    f"🧠 Generate {explanation_method} Explanation",
                    type="primary",
            ):

                if explanation_method == "LIME":

                    with st.spinner("Generating LIME explanation..."):
                        explanation = explain_text(selected_text)

                    explanation_list = explanation.as_list(
                        label=(
                            1 if selected_prediction == "Cyberbullying"
                            else 0
                        )
                    )

                    explanation_df = pd.DataFrame(
                        explanation_list,
                        columns=["Token", "Contribution"],
                    )

                    st.markdown("### 🔬 LIME Explanation")
                    st.dataframe(
                        explanation_df,
                        use_container_width=True,
                        hide_index=True,
                    )

                else:

                    with st.spinner("Generating SHAP explanation..."):
                        shap_exp, predicted_class, probabilities = (
                            explain_with_shap(selected_text)
                        )

                    shap_df = pd.DataFrame({
                        "Token": list(shap_exp.data),
                        "SHAP Value": np.asarray(
                            shap_exp.values
                        ).flatten(),
                    })

                    shap_df = shap_df[
                        shap_df["Token"].astype(str).str.strip() != ""
                        ]

                    shap_df = shap_df.sort_values(
                        "SHAP Value",
                        key=lambda x: x.abs(),
                        ascending=False,
                    )

                    st.markdown("### 🧠 SHAP Explanation")

                    st.write(
                        f"Explaining prediction: **{CLASS_NAMES[predicted_class]}**"
                    )

                    st.dataframe(
                        shap_df.reset_index(drop=True),
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.markdown("#### Token Importance")

                    if not shap_df.empty:
                        st.bar_chart(
                            shap_df.head(15)
                            .sort_values("SHAP Value")
                            .set_index("Token")["SHAP Value"]
                        )
        except Exception as e:

            st.error(
                "Unable to process the uploaded CSV file."
            )
            st.exception(e)

            st.caption(str(e))