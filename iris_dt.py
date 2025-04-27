from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import os

# Load data
iris = load_iris()
X = iris.data
y = iris.target

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Hyperparameter
max_depth = 2

# Set experiment
mlflow.set_experiment('iris_dt')

# Set the tracking URI explicitly to a local directory
mlflow.set_tracking_uri("http://127.0.0.1:5000/")

with mlflow.start_run():
    # Train model
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Log metrics and params
    mlflow.log_metric('accuracy', accuracy)
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

    # Save and log to MLflow
    heatmap_path = 'confusion_matrix_heatmap.png'
    plt.savefig(heatmap_path)
    plt.close()
    #log plot
    mlflow.log_artifact(heatmap_path)
    
    #log current code
    mlflow.log_artifact(__file__)
    
    #log model
    mlflow.sklearn.log_model(dt,'Decision Tree')
    
    mlflow.set_tag('author','PrashantJ')
    mlflow.set_tag('model','DT')
    

    print('Accuracy:', accuracy)