# Premier League Match Prediction App

A machine learning–based web application that predicts the outcome of English Premier League matches (Home Win, Draw, Away Win) using historical match data, advanced feature engineering, FIFA team ratings, and an XGBoost classification model. The application is built with Streamlit and provides an interactive, visually rich user interface inspired by modern football analytics tools.

---

## Project Overview

This project models football match outcomes by combining multiple data sources and feature sets, including:

* Historical Premier League match results
* Team form and performance trends
* Elo ratings
* Rest days between matches
* Head-to-head statistics
* Bookmaker implied probabilities
* FIFA team ratings (attack, midfield, defence, overall)

An XGBoost classifier is trained on engineered pre-match features to estimate probabilities for each possible result. The trained model is then deployed in a Streamlit web application for real-time predictions.

---

## Application Interface

### Match Selection and Team Comparison

The app allows users to:

* Select home and away teams using navigation buttons
* View official club logos
* Compare FIFA-based Attack, Midfield, and Defence ratings
* Trigger predictions with a single click

### Prediction Output

Once a prediction is made, the app displays:

* Predicted match result (Home Win / Draw / Away Win)
* Probability distribution across all outcomes
* A detailed table of features used for the prediction

---

## Machine Learning Model

* **Model Type:** XGBoost Classifier
* **Target Variable:** Full-Time Result (FTR)

  * Home Win → 2
  * Draw → 1
  * Away Win → 0
* **Model File:** `models/xgb_model.joblib`
* **Features:** 42 engineered pre-match features

### Why XGBoost?

XGBoost was chosen due to:

* Strong performance on structured/tabular data
* Ability to model non-linear feature interactions
* Robust handling of feature importance and class imbalance
* Proven success in sports analytics tasks

---

## Feature Engineering

The project emphasizes strict prevention of data leakage by ensuring all features are computed using only information available before each match.

Key engineered features include:

### Team Form

* Rolling points per game
* Rolling goals scored and conceded
* Win and loss streaks

### Match Context

* Home and Away Elo ratings
* Rest days since last match

### Head-to-Head Statistics

* Recent wins between teams
* Goals scored in past encounters
* Number of recent meetings

### Bookmaker Odds

* Implied probabilities for home, draw, and away outcomes
* Normalized odds-based features

### FIFA Ratings

* Attack, Midfield, Defence, Overall ratings
* Home vs Away rating differences

---

## Project Structure

```
Football-Prediction-App/
│
├── app.py                     # Streamlit application
├── pipeline.py                # Data cleaning & feature engineering pipeline
├── models/
│   └── xgb_model.joblib       # Trained XGBoost model
│
├── data/
│   ├── pl-22.csv
│   ├── pl-23.csv
│   ├── pl-24.csv
│   ├── pl-25.csv
│   └── tbl_team.csv           # FIFA team ratings
│
├── logos/                     # Club logos
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Technologies Used

* Python
* Streamlit
* XGBoost
* Pandas, NumPy
* Scikit-learn
* Joblib (model serialization)

---

## Installation and Usage

### 1. Clone the Repository

```bash
git clone https://github.com/sahil-kulavoor/Football-Prediction-App.git
cd Football-Prediction-App
```

### 2. Create and Activate Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Known Limitations

These are documented honestly so the results are interpreted correctly:

1. **Bookmaker odds are unavailable at prediction time.** The three implied-probability
   features account for roughly 22% of the model's total gain (they are the two
   most important features overall), but a future fixture has no odds attached, so
   they are passed to the model as zero. Predicted probabilities are therefore
   flatter and less reliable than the training metrics suggest. Adding odds inputs
   to the UI, or retraining without odds features, would resolve this.

2. **The bundled FIFA ratings table is dated.** It does not contain Bournemouth,
   Brentford, Ipswich, Luton, Nott'm Forest or Sunderland, so those clubs show no
   ratings and contribute zeros for 12 of the 42 features.

3. **The shipped model was trained before the date-parsing fix** described in the
   commit history, so it was fitted on a partially corrupted view of the data.
   Retraining on the corrected dataset is recommended.

---

## Academic and Practical Relevance

This project demonstrates:

* End-to-end machine learning pipeline design
* Safe feature engineering for time-series sports data
* Deployment of ML models in interactive web applications
* Integration of external rating systems (FIFA) into predictive modeling

It is suitable for academic coursework, machine learning portfolios, and applied sports analytics research.

---

## Future Improvements

* Expected goals (xG)–based features
* Player-level injury and lineup data
* Probability calibration visualizations
* Multi-league support
* Model comparison with neural networks or Bayesian approaches

---

## Author

Developed by **Sahil Kulavoor**
M.Tech Artificial Intelligence
Premier League Match Prediction System

---
