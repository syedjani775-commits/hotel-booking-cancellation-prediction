import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

def run_evaluation(model_name="random_forest"):
    test_df = pd.read_csv("data/processed/test.csv")
    X_test = test_df.drop(columns="is_canceled")
    y_test = test_df["is_canceled"]

    model_path = f"models/{model_name}.pkl"
    pipe = joblib.load(model_path)

    preds = pipe.predict(X_test)

    print(f"=== Evaluation Report for {model_name} ===")
    print(classification_report(y_test, preds))
    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, preds))

if __name__ == "__main__":
    run_evaluation("random_forest")
