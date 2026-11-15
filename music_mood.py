import argparse

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TRAINING_FEATURES = np.array([
    [70, 0.20], [82, 0.30], [90, 0.25], [105, 0.40],
    [118, 0.65], [128, 0.80], [140, 0.90], [150, 0.95],
    [60, 0.15], [100, 0.50], [135, 0.85], [75, 0.28],
], dtype=float)
TRAINING_LABELS = np.array([
    "calm", "calm", "calm", "calm", "energetic", "energetic",
    "energetic", "energetic", "calm", "energetic", "energetic", "calm",
])


def build_model() -> KNeighborsClassifier:
    model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3))
    model.fit(TRAINING_FEATURES, TRAINING_LABELS)
    return model


def predict_mood(tempo: float, energy: float) -> str:
    if tempo <= 0 or not 0 <= energy <= 1:
        raise ValueError("Tempo must be positive and energy must be between 0 and 1")
    return str(build_model().predict([[tempo, energy]])[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify a song's mood.")
    parser.add_argument("--tempo", type=float, required=True, help="Tempo in beats per minute")
    parser.add_argument("--energy", type=float, required=True, help="Energy from 0 to 1")
    args = parser.parse_args()
    print(f"Predicted mood: {predict_mood(args.tempo, args.energy)}")


if __name__ == "__main__":
    main()
