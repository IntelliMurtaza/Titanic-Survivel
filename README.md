```markdown
# Titanic Survival Predictor

A machine learning project that predicts passenger survival on the Titanic.  
Built with logistic regression, random forest, and XGBoost, featuring automated hyperparameter tuning and a Streamlit web application for live predictions.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Exploratory Data Analysis & Cleaning](#exploratory-data-analysis--cleaning)
- [Feature Engineering](#feature-engineering)
- [Modeling Approach](#modeling-approach)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Results & Model Evaluation](#results--model-evaluation)
- [Final Model](#final-model)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## Overview

This project applies the end‑to‑end machine learning workflow to the classic **Titanic – Machine Learning from Disaster** competition dataset. The goal is to accurately predict which passengers survived the sinking based on demographic and ticket information.

**Workflow steps:**

1. Exploratory Data Analysis (EDA)  
2. Data cleaning (handling missing values, outliers)  
3. Feature engineering (creating new informative features)  
4. Model training, comparison, and selection  
5. Hyperparameter tuning  
6. Evaluation with detailed metrics  
7. Deployment of the selected model via a Streamlit app

---

## Dataset

- **Source:** [Kaggle – Titanic Dataset](https://www.kaggle.com/c/titanic)  
- **Target column:** `Survived` (0 = No, 1 = Yes)  
- **Important features used:**
  - `Pclass` – Passenger class (1st, 2nd, 3rd)
  - `Sex`
  - `Age`
  - `SibSp` – Number of siblings/spouses aboard
  - `Parch` – Number of parents/children aboard
  - `Fare`
  - `Embarked` – Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)
  - `Cabin` (partial information used to extract `Deck`)

---

## Exploratory Data Analysis & Cleaning

- Checked for missing values: `Age`, `Cabin`, and `Embarked` had missing entries.
- Visualised distributions, correlations, and survival rates across categorical variables (e.g., sex, class).
- **Cleaning steps:**
  - `Age` missing values imputed with the **median** age grouped by `Pclass` and `Sex`.
  - `Embarked` missing values filled with the **mode**.
  - `Cabin` used to extract `Deck`; missing cabins left as `Unknown`.
  - `Fare` checked for skewness – no extreme outliers required removal.

---

## Feature Engineering

To improve model accuracy, several new features were created:

| Feature       | Description                                                                 |
|---------------|-----------------------------------------------------------------------------|
| `FamilySize`  | `SibSp + Parch + 1` (total family members aboard)                           |
| `IsAlone`     | Binary: 1 if `FamilySize == 1`, else 0                                     |
| `Title`       | Extracted from the `Name` field (e.g., Mr, Mrs, Miss, Master, etc.)        |
| `Deck`        | First character of the `Cabin` string; `Unknown` if cabin is missing       |

All feature engineering was performed **after splitting the data into train/test sets** to avoid data leakage.

---

## Modeling Approach

The data was split into **80% training** and **20% testing**, stratified by the target variable. Three classifiers were trained and compared:

- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier

Categorical features were encoded using one‑hot encoding or label encoding, and numerical features were scaled where necessary (e.g., for logistic regression).

---

## Hyperparameter Tuning

Two search strategies were applied on the training set using **5‑fold cross‑validation**:

- **RandomizedSearchCV** – wide, random exploration of hyperparameter space  
- **GridSearchCV** – fine‑grained search around the best candidates from the random search

The best parameters for each model were selected based on **accuracy**. (Future iterations can optimise for F1‑score to better handle class imbalance.)

---

## Results & Model Evaluation

Accuracy on the held‑out test set:

| Model               | Accuracy |
|---------------------|----------|
| Logistic Regression | **82%**  |
| Random Forest       | 80%      |
| XGBoost             | 81%      |

Because the dataset is slightly imbalanced (~38% survived), accuracy alone can be misleading. Therefore, the **final model was evaluated in detail**:

### Logistic Regression – Classification Report

```
              precision    recall  f1-score   support
           0       0.83      0.86      0.84       105
           1       0.79      0.75      0.77        74

    accuracy                           0.82       179
   macro avg       0.81      0.81      0.81       179
weighted avg       0.81      0.82      0.81       179
```

**Confusion Matrix:**
```
                Predicted No   Predicted Yes
Actual No            90              15
Actual Yes           18              56
```

The model shows balanced performance across both classes, making it a reliable choice.

---

## Final Model

**Logistic Regression** was chosen as the final model due to its top accuracy, simplicity, and strong interpretability. The trained model is saved in the `models/` directory and served through the Streamlit app for real‑time predictions.

---

## Project Structure

```
Titanic-Survival-Predictor/
│
├── data/                   # Raw and processed dataset files
├── notebooks/              # Jupyter notebooks for EDA, modelling, evaluation
├── src/                    # Python scripts for preprocessing, training, etc.
├── models/                 # Serialised model (.pkl) and scalers/encoders
├── app.py                  # Streamlit web application
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── LICENSE                 # (optional) License file
```

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/titanic-survival-predictor.git
   cd titanic-survival-predictor
   ```

2. (Optional) Create and activate a virtual environment.

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

**Requirements (example):**
```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
streamlit
```

---

## Usage

### Run the Streamlit Web App

```bash
streamlit run app.py
```

The app allows you to input passenger details and instantly get a survival prediction from the trained Logistic Regression model.

### Retrain or Experiment

Explore the Jupyter notebooks inside `notebooks/` to see the step‑by‑step analysis, feature engineering, and model tuning.  
Run the scripts in `src/` to reproduce the entire pipeline from raw data to trained model.

---

## Future Improvements

- Implement **Stratified K‑Fold cross‑validation** and report mean ± std metrics for robustness.
- Experiment with **ensemble methods** (Voting Classifier, Stacking).
- Perform **more advanced feature engineering** (e.g., fare per person, interaction terms).
- Add **hyperparameter optimisation** using F1‑score to better handle class imbalance.
- Integrate **SHAP** or **LIME** for model interpretability.
- Containerise the app using Docker for easier deployment.
- Set up CI/CD to automatically test and deploy the app.

---

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.  
(You can add one if you wish.)

---

*Built as a classic machine learning portfolio project.*
```
