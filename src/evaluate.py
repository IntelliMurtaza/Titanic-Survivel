import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from utils import load_data, load_model

def evaluate_model():
    print("Ingesting raw data titanic.csv dataset for evaluation...")
    df = load_data("../data/titanic.csv")
    print("Data ingested successfully.")

    # Extract only the exact raw columns needed
    features = ['Age', 'Fare', 'SibSp', 'Parch', 'Sex', 'Embarked']
    X = df[features]
    y = df['Survived']
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Load the trained model
    model_pipeline = load_model("../models/logistic_regression_pipeline.pkl")
    # Make predictions
    y_pred = model_pipeline.predict(X_test)


    # Calculate evaluation metrics
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Model evaluation completed successfully.")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

    #Generate and save a confusion image file
    print("Generating confusion matrix plot ...")
    ConfusionMatrixDisplay.from_estimator(model_pipeline, X_test, y_test, cmap=plt.cm.Blues)
    plt.title("Model Evaluation Matrix")
    plt.savefig("../results/confusion_matrix.png")
    plt.close()

if __name__ == "__main__":
    evaluate_model()