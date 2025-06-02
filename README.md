# Click-Through Rate Prediction using XGBoost

This project tackles the **Click-Through Rate (CTR) Prediction** problem using machine learning, specifically the **XGBoost classifier**. The dataset comes from the [Kaggle CTR Prediction Competition](https://www.kaggle.com/c/avazu-ctr-prediction/overview) sponsored by **Avazu**, which provides ad click logs over 10 days. The objective is to **predict the probability** that a mobile ad will be clicked.


## Dataset

- **Source**: [Kaggle Avazu CTR Prediction](https://www.kaggle.com/c/avazu-ctr-prediction)
- **Data**: 10 days of click-through log data
- **Target variable**: `click` (1 = clicked, 0 = not clicked)
- **Sample used**: Due to computation constraints, a **random 100,000-row sample** of the full dataset is used for this project.


## Model

We use **XGBoost**, a powerful gradient boosting algorithm, to train a binary classifier for predicting ad clicks.

### Key Features of the Model:
- Extensive **feature engineering** (e.g., timestamp features, categorical encodings)
- Handling **class imbalance**
- **Cross-validation** using ROC AUC as the primary scoring metric
- **Hyperparameter tuning** with `GridSearchCV`
- Visualization of performance metrics and feature importance


## Evaluation Metrics

The model performance is primarily measured using:

- **ROC AUC Score**: Evaluates model's ability to separate clicked vs. non-clicked ads
- **Log Loss**: Evaluates the accuracy of predicted probabilities
- **Precision / Recall / F1 Score** (from classification reports)



## Tools & Libraries

- Python (3.8+)
- pandas, numpy
- scikit-learn
- XGBoost
- seaborn, matplotlib


## How to Run

1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/ctr-prediction-xgboost.git
    cd ctr-prediction-xgboost
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Jupyter notebook or Python script:
    ```bash
    jupyter notebook CTR_Prediction_XGBoost.ipynb
    ```

---

## 📌 Project Structure

