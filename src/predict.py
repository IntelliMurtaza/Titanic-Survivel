import pandas as pd
from utils import load_data, load_model
from train import run_training

def make_predictions():
    #print("Runnig prediction pipeline on new test data...")
    # Load real unseen test data
    #new_data = load_data("../data/titanic.csv")
    
    # Keep required subset features for prediction
    #X_test = new_data[['Age', 'Fare', 'SibSp', 'Parch', 'Sex', 'Embarked']]
    X_test, y_test = run_training()  # Get the test data from the training function

    # Load fitted pipeline
    model_pipeline = load_model("../models/logistic_regression_pipeline.pkl")  


    # Make predictions
    predictions = model_pipeline.predict(X_test)
    probabilities = model_pipeline.predict_proba(X_test)[:, 1]  # Probability of survival

    # Create a DataFrame with the predictions and probabilities
    ''' results = pd.DataFrame({
        'PassengerId': new_data['PassengerId'],
        'Survived': predictions,
        'SurvivalProbability': probabilities
    })

    # Save the results to a CSV file
    results.to_csv("results/predictions.csv", index=False)
    print("Predictions saved to results/predictions.csv")'''

if __name__ == "__main__":
    make_predictions()