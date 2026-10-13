# Cyberbullying Detection

An explainable AI application for detecting cyberbullying in text comments using a fine-tuned multilingual BERT (mBERT) model. Built with Python and Streamlit, the application supports individual comment analysis, batch CSV processing, and model explanations using LIME and SHAP.

## Overview

Cyberbullying can occur across social media platforms, discussion forums, and online communities. This project uses Natural Language Processing (NLP) and a fine-tuned mBERT model to classify text comments into two categories:

- **Cyberbullying**
- **Non-Cyberbullying**

The application also provides explainability features to help users understand which words contribute to a model prediction.

## Features

- **Single Comment Detection:** Analyze an individual comment and obtain its predicted category and confidence.
- **Batch Detection:** Upload a CSV file containing multiple comments for classification.
- **LIME Explainability:** Identify words that contribute to a prediction using Local Interpretable Model-agnostic Explanations.
- **SHAP Explainability:** Analyze word-level contributions using SHapley Additive exPlanations.
- **Interactive Interface:** Use a Streamlit web interface to access the detection and explanation features.
- **Prediction Probabilities:** View the model's probability estimates for the available classes.

## Technology Stack

- **Language:** Python 3.11
- **NLP Model:** Fine-tuned multilingual BERT (mBERT)
- **Machine Learning:** PyTorch, Hugging Face Transformers
- **Web Interface:** Streamlit
- **Explainable AI:** LIME, SHAP
- **Data Processing:** Pandas, NumPy

## Project Structure

```text
cyberbullying-detection/
├── app.py
├── backend/
│   ├── predictor.py
│   ├── explainer.py
│   └── shap_explainer.py
├── pages/
│   └── 3_Explainability.py
├── model/                  # Trained model files (not tracked by Git)
├── requirements.txt
├── test_model.py
├── .gitignore
└── README.md
```

The exact files inside `backend/` and `pages/` may vary as the project evolves.

## Prerequisites

- Python 3.11.x
- Git
- pip
- A compatible computer with sufficient memory to load the model

Python 3.11.6 and Python 3.11.9 are both suitable starting points for this project.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/cgiop/cyberbullying-detection.git
cd cyberbullying-detection
```

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Obtain the trained model

The trained model files are excluded from this repository to keep the Git repository size manageable.

Obtain the model files from the project owner or the approved shared location. Place them inside the `model/` directory while preserving the expected filenames and folder structure.

**Note:** The application may not run predictions until the trained model files are available locally.

### 5. Run the application

From the project root, run:

```powershell
streamlit run app.py --server.fileWatcherType none
```

Streamlit will provide a local URL, usually:

```text
http://localhost:8501
```

Open that URL in your browser.

## Using the Application

### Single Comment Detection

1. Open the application.
2. Enter a text comment.
3. Run the detection process.
4. Review the predicted class and confidence or probability estimates.

### Batch Detection

1. Prepare a CSV file containing a column named `text`.
2. Enter one comment per row.
3. Upload the CSV through the batch detection interface.
4. Review the predictions and summary results.

Example CSV:

```csv
text
"You are very kind and helpful."
"Nobody wants you here. Leave us alone."
```

These examples are illustrative and do not guarantee particular model predictions.

### Explainability

1. Open the Explainable AI page.
2. Choose a single comment or upload a CSV.
3. Select **LIME** or **SHAP**.
4. Generate an explanation for the selected comment.
5. Review words supporting or opposing the prediction.

LIME and SHAP use different explanation methods. Their values should not be compared directly, and highlighted words should not be treated as universal indicators of cyberbullying.

## Model Information

The application uses a fine-tuned mBERT classifier based on `bert-base-multilingual-cased`.

The classifier produces predictions for two classes:

| Class ID | Label |
|---|---|
| 0 | Non-Cyberbullying |
| 1 | Cyberbullying |

The model's predictions depend on its training data and learned patterns. The predicted class and confidence should not be treated as definitive proof that a comment is or is not cyberbullying.

## Testing

The repository includes `test_model.py` for model-related testing.

After installing the dependencies and placing the model files in the expected location, run:

```powershell
python test_model.py
```

The test script may require the trained model files to be present.

## Limitations

- Model performance depends on the quality, diversity, and representativeness of the training data.
- Context, sarcasm, slang, and ambiguous language can affect classification.
- Predictions can contain false positives and false negatives.
- LIME and SHAP explanations describe model behavior; they do not prove that a prediction is correct or establish causation.
- This project is intended as an educational and research application, not as a replacement for human moderation or contextual review.

## Future Improvements

- Evaluate the model on additional datasets and unseen comments.
- Improve robustness to slang, sarcasm, and context-dependent language.
- Add further performance metrics and error analysis.
- Improve batch reporting and visualization.
- Explore deployment options for wider access.

## Contributing

Contributions, suggestions, and bug reports are welcome. For collaborative development, create a branch for your changes and open a pull request for review.

## License

No license has been specified yet. Unless a license is added, reuse and redistribution are subject to applicable copyright restrictions.

## Acknowledgements

This project uses the Hugging Face Transformers ecosystem, Streamlit, LIME, SHAP, PyTorch, and other open-source Python libraries.
