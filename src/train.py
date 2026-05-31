from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from preprocessing import get_preprocessing_pipeline
from utils import load_data, save_model

def run_training():
    print("Ingesting raw data train.csv dataset...")
    df = load_data("../data/titanic.csv")
    print("Data ingested successfully.")

    # Extract only the exact raw columns needed
    features = ['Age', 'Fare', 'SibSp', 'Parch', 'Sex', 'Embarked']
    X = df[features]
    y = df['Survived']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Get the preprocessing pipeline
    preprocessor = get_preprocessing_pipeline()

    # Create the full pipeline
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42))
    ])

    # Train the model
    print('fitting the master pipeline with clean weights...')
    full_pipeline.fit(X_train, y_train)

    save_model(full_pipeline, "../models/logistic_regression_pipeline.pkl")
    print("Model training and saving completed successfully.")

    return X_test, y_test

if __name__ == "__main__":
    run_training()

