from lime.lime_text import LimeTextExplainer

from backend.predictor import predict_proba


explainer = LimeTextExplainer(
    class_names=[
        "Non-Cyberbullying",
        "Cyberbullying"
    ]
)


def explain_text(text, num_samples=100):
    """
    Generate a LIME explanation for the given text.
    """

    explanation = explainer.explain_instance(
        text,
        predict_proba,
        labels=(0,1),
        num_features=10,
        num_samples=num_samples
    )

    return explanation