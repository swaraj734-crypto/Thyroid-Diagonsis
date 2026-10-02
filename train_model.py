
import argparse
import joblib
from sklearn.model_selection import train_test_split

from src.data_utils import load_dataset, encode_features
from src.hybrid_model import ThyroidHybridModel


def main(data_path: str, model_dir: str):
    print(f"Loading dataset from {data_path} ...")
    df = load_dataset(data_path)
    X, y, encoders = encode_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training CatBoost + ANN hybrid ensemble ...")
    model = ThyroidHybridModel(catboost_weight=0.6)
    model.fit(X_train.values, y_train)

    preds = model.predict(X_test.values)
    acc = (preds == y_test).mean()
    print(f"Hold-out accuracy: {acc:.3f}")

    print(f"Saving model artifacts to {model_dir}/ ...")
    model.save(model_dir)
    joblib.dump(encoders, f"{model_dir}/encoders.pkl")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sample_thyroid_data.csv")
    parser.add_argument("--model_dir", default="models")
    args = parser.parse_args()

