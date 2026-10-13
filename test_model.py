from backend.predictor import predict_cyberbullying


test_texts = [
    "You are doing a great job!",
    "Nobody likes you, you are completely useless.",
    "Have a wonderful day.",
    "Go away idiot, nobody wants you here."
]


for text in test_texts:

    result = predict_cyberbullying(text)

    print("\n" + "=" * 60)

    print("Text:")
    print(text)

    print("\nPrediction:")
    print(result["label"])

    print("\nConfidence:")
    print(f"{result['confidence']:.2%}")

    print("\nProbabilities:")
    print(
        f"Non-Cyberbullying: "
        f"{result['probabilities']['Non-Cyberbullying']:.2%}"
    )

    print(
        f"Cyberbullying: "
        f"{result['probabilities']['Cyberbullying']:.2%}"
    )