# Titanic Survival Predictor

## Project Overview

This project predicts whether a passenger survived the Titanic disaster using machine learning techniques.

The project follows a complete machine learning workflow, including:

* Exploratory Data Analysis (EDA)
* Data Cleaning
* Feature Engineering
* Model Comparison and Selection
* Hyperparameter Tuning
* Model Evaluation

The final model was deployed using Streamlit for interactive predictions.

---

## Dataset

**Dataset:** Titanic Dataset

**Source:** Kaggle

**Target Variable:**

* Survived

**Important Features:**

* Age
* Sex
* Fare
* Pclass
* Embarked
* Parch
* SibSp

---

## Feature Engineering

Several new features were created to improve model performance:

### Created Features

* **FamilySize** = SibSp + Parch + 1
* **IsAlone** = 1 if FamilySize = 1, otherwise 0
* **Title** extracted from the passenger's name
* **Deck** extracted from the Cabin feature

### Missing Value Handling

* Numerical features were imputed using the mean value.
* Categorical features were imputed using the mode value.

---

## Models Evaluated

The following machine learning models were tested:

1. Logistic Regression
2. Random Forest Classifier
3. XGBoost Classifier

---

## Hyperparameter Tuning

The following techniques were used for hyperparameter optimization:

* RandomizedSearchCV
* GridSearchCV

---

## Results

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | 82%      |
| Random Forest       | 80%      |
| XGBoost             | 81%      |

### Final Selected Model

**Logistic Regression**

---

## Project Structure

```text
Project/
│
├── data/
├── notebooks/
├── models/
├── src/
├── app.py
├── README.md
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

---

## Future Improvements

* Add cross-validation reporting
* Experiment with ensemble methods
* Improve feature engineering
* Deploy the application to a cloud platform
* Add additional evaluation metrics such as Precision, Recall, F1-Score, and ROC-AUC

---

## Author

Murtaza

This project was created as part of my machine learning machine learning engineering portfolio project to practice data preprocessing, feature engineering, model selection, hyperparameter tuning, and deployment.
