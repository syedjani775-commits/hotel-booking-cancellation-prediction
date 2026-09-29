import mlflow
import mlflow.sklearn
import pandas as pd
import joblib
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

def track_experiments():
    mlflow.set_experiment("Hotel_Booking_Cancellation")
    
    test_df = pd.read_csv("data/processed/test.csv")
    X_test, y_test = test_df.drop(columns="is_canceled"), test_df["is_canceled"]
    
    models = ["logistic_regression", "decision_tree", "random_forest"]
    
    for m_name in models:
        with mlflow.start_run(run_name=m_name):
            pipeline = joblib.load(f"models/{m_name}.pkl")
            preds = pipeline.predict(X_test)
            probs = pipeline.predict_proba(X_test)[:, 1]
            
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds)
            auc = roc_auc_score(y_test, probs)
            
            # Log params & metrics (Experiment 4)
            mlflow.log_param("model_name", m_name)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc", auc)
            
            # Register candidate model with pickle serialization (Experiment 6)
            reg_name = "Hotel_Cancellation_Production" if m_name == "random_forest" else None
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                serialization_format="pickle",
                registered_model_name=reg_name
            )
            print(f"Logged {m_name} to MLflow. F1 Score: {f1:.4f}")

if __name__ == "__main__":
    track_experiments()
