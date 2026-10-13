import streamlit as st
import pandas as pd

from backend.predictor import predict_cyberbullying


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Batch Detection",
    page_icon="📊",
    layout="wide"
)


st.title("📊 Batch Cyberbullying Detection")

st.write(
    "Upload a CSV file containing social media comments "
    "and classify them using the trained mBERT model."
)


# --------------------------------------------------
# File upload
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"]
)


if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")

    st.table(df)

    # --------------------------------------------------
    # Check text column
    # --------------------------------------------------

    if "text" not in df.columns:

        st.error(
            "Your CSV must contain a column named 'text'."
        )

    else:

        if st.button(
            "🔍 Analyze All Comments",
            type="primary"
        ):

            predictions = []
            confidences = []

            progress_bar = st.progress(0)

            total = len(df)

            for i, text in enumerate(df["text"]):

                # Handle missing values
                if pd.isna(text):

                    text = ""

                text = str(text)

                result = predict_cyberbullying(text)

                predictions.append(
                    result["label"]
                )

                confidences.append(
                    result["confidence"]
                )

                progress_bar.progress(
                    (i + 1) / total
                )


            # --------------------------------------------------
            # Add predictions to dataframe
            # --------------------------------------------------

            df["Prediction"] = predictions

            df["Confidence"] = confidences


            st.success(
                f"Successfully analyzed {total} comments."
            )


            # --------------------------------------------------
            # Results
            # --------------------------------------------------

            st.subheader("Detection Results")

            display_df = df.copy()

            display_df["Confidence"] = (
                display_df["Confidence"] * 100
            ).round(2).astype(str) + "%"


            st.table(display_df)


            # --------------------------------------------------
            # Statistics
            # --------------------------------------------------

            cyberbullying_count = (
                df["Prediction"] == "Cyberbullying"
            ).sum()

            normal_count = (
                df["Prediction"] == "Non-Cyberbullying"
            ).sum()


            # --------------------------------------------------
            # Metrics
            # --------------------------------------------------

            col1, col2, col3, col4 = st.columns(4)


            with col1:

                st.metric(
                    "Total Comments",
                    total
                )


            with col2:

                st.metric(
                    "Cyberbullying",
                    cyberbullying_count
                )


            with col3:

                st.metric(
                    "Non-Cyberbullying",
                    normal_count
                )


            with col4:

                cyberbullying_percentage = (
                    cyberbullying_count / total * 100
                    if total > 0 else 0
                )

                st.metric(
                    "Cyberbullying Rate",
                    f"{cyberbullying_percentage:.1f}%"
                )


            # --------------------------------------------------
            # Visualization
            # --------------------------------------------------

            st.subheader("📊 Detection Summary")


            chart_data = pd.DataFrame(
                {
                    "Prediction": [
                        "Cyberbullying",
                        "Non-Cyberbullying"
                    ],
                    "Count": [
                        cyberbullying_count,
                        normal_count
                    ]
                }
            )

            st.bar_chart(
                chart_data.set_index("Prediction")
            )


            # --------------------------------------------------
            # Confidence Analysis
            # --------------------------------------------------

            st.subheader("🎯 Confidence Analysis")


            average_confidence = df["Confidence"].mean()


            st.write(
                f"Average model confidence: "
                f"**{average_confidence:.2%}**"
            )


            # --------------------------------------------------
            # Download results
            # --------------------------------------------------

            csv_data = df.to_csv(
                index=False
            ).encode("utf-8")


            st.download_button(
                label="📥 Download Results",
                data=csv_data,
                file_name="cyberbullying_predictions.csv",
                mime="text/csv"
            )
try:
    df = pd.read_csv(uploaded_file)

    if df.empty:
        st.error("The uploaded CSV file is empty.")
        st.stop()

    if "text" not in df.columns:
        st.error(
            "The CSV file must contain a column named 'text'."
        )
        st.write("Columns found:", list(df.columns))
        st.stop()

    # Remove rows with missing text
    df = df.dropna(subset=["text"])

    # Convert text column to string
    df["text"] = df["text"].astype(str)

    # Remove empty comments
    df = df[df["text"].str.strip() != ""]

    if df.empty:
        st.error(
            "No valid comments were found in the 'text' column."
        )
        st.stop()

except Exception as e:
    st.error("Unable to read the uploaded CSV file.")
    st.caption(str(e))
    st.stop()