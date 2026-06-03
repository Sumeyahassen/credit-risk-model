import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def train_models():
    # Load data
    df = pd.read_csv('data/processed/processed_data.csv')
    
    # Features and Target
    feature_cols = ['Recency', 'Frequency', 'Monetary', 'AvgValue', 
                   'StdValue', 'MaxValue', 'AvgAmount']
    
    X = df[feature_cols]
    y = df['is_high_risk']
    
    # ====================== FIX NaNs ======================
    imputer = SimpleImputer(strategy='median')
    X = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols)
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # Models
    models = {
        "Logistic_Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random_Forest": RandomForestClassifier(n_estimators=200, random_state=42)
    }
    
    mlflow.set_experiment("Bati_Bank_Credit_Risk")
    
    print("🚀 Starting Model Training with NaN Handling...\n")
    
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_prob)
            
            # Log to MLflow
            mlflow.log_metric("accuracy", accuracy) # type: ignore
            mlflow.log_metric("precision", precision) # type: ignore
            mlflow.log_metric("recall", recall) # type: ignore
            mlflow.log_metric("f1_score", f1) # type: ignore
            mlflow.log_metric("roc_auc", roc_auc) # type: ignore
            
            mlflow.sklearn.log_model(model, f"{name}_model") # type: ignore
            
            # Print
            print(f"✅ {name} Completed")
            print(f"   ROC-AUC  : {roc_auc:.4f}")
            print(f"   F1 Score : {f1:.4f}")
            print(f"   Precision: {precision:.4f}")
            print(f"   Recall   : {recall:.4f}")
            print(f"   Accuracy : {accuracy:.4f}")
            print("-" * 55)

if __name__ == "__main__":
    train_models()