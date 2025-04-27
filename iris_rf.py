from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import os
import dagshub

# Init dagshub
dagshub.init(repo_owner='iamprashantjain', repo_name='mlflow-dagshub-demo', mlflow=True)

# Set MLflow tracking URI (dagshub)
mlflow.set_tracking_uri("https://dagshub.com/iamprashantjain/mlflow-dagshub-demo.mlflow")

# Load data
iris = load_iris()
X = iris.data
y = iris.target

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Hyperparameters
n_estimators = 100
max_depth = 10

# Set experiment
mlflow.set_experiment('iris_rf')

with mlflow.start_run(run_name='RandomForest_maxdepth10'):
    # Train Random Forest model
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Log metrics and params
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_param('n_estimators', n_estimators)
    mlflow.log_param('max_depth', max_depth)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=iris.target_names, columns=iris.target_names)

    # Plot heatmap
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix Heatmap')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    # Save and log heatmap
    heatmap_path = 'confusion_matrix_heatmap_rf.png'
    plt.savefig(heatmap_path)
    plt.close()
    mlflow.log_artifact(heatmap_path)

    # Log current script (if not run as Jupyter)
    try:
        mlflow.log_artifact(__file__)
    except:
        pass  # Ignore in Jupyter or notebooks

    # Log model
    mlflow.sklearn.log_model(rf, 'RandomForestModel')

    # Set tags
    mlflow.set_tag('author', 'PrashantJ')
    mlflow.set_tag('model', 'RandomForest')

    print('Accuracy:', accuracy)