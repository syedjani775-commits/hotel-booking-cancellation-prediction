import os
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

def build_preprocessor(X):
    num_cols = X.select_dtypes(exclude="object").columns.tolist()
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])

def train_baselines():
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    
    X_train, y_train = train_df.drop(columns="is_canceled"), train_df["is_canceled"]
    X_test, y_test = test_df.drop(columns="is_canceled"), test_df["is_canceled"]
    
    preprocessor = build_preprocessor(X_train)
    
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "decision_tree": DecisionTreeClassifier(max_depth=12, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=1)
    }
    
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    summary = []
    for name, clf in models.items():
        print(f"Training {name}...")
        pipe = Pipeline([("prep", preprocessor), ("model", clf)])
        pipe.fit(X_train, y_train)
        
        preds = pipe.predict(X_test)
        probs = pipe.predict_proba(X_test)[:, 1]
        
        f1 = f1_score(y_test, preds)
        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        
        summary.append({"Model": name, "Accuracy": round(acc, 4), "F1": round(f1, 4), "ROC_AUC": round(auc, 4)})
        joblib.dump(pipe, f"models/{name}.pkl")
        
    df_results = pd.DataFrame(summary)
    df_results.to_csv("outputs/baseline_metrics.csv", index=False)
    print("\nBaseline Training Complete:")
    print(df_results)

if __name__ == "__main__":
    train_baselines()
