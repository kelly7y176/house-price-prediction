# 🏡 King County House Price Prediction Pipeline

This project implements an end-to-end Machine Learning pipeline to predict residential house prices in King County, USA. </br>
It covers everything from data preprocessing and feature engineering to model deployment via a web interface.

## 🚀 Live Demo
You can interact with the deployed model here: 
 [➡️ Launch King County House Price Prediction Pipeline](https://house-price-prediction-9roq5xt4fubsjvuujt4kxw.streamlit.app/)

<!-- Cover -->
<br />
<div>
   <img align="left" align="left" alt="HTML" width="100%" style="padding-right:10px;" src="https://i.imgur.com/2u67gYU.jpeg" />   
</div>

---

## 📊 Project Overview
The goal of this assignment was to build a robust regression model capable of estimating property values based on structural and geographical data.

### 🛠️ The Pipeline
1. **Data Cleaning**: Handled missing values via median imputation and removed non-predictive metadata (`id`, `date`).
2. **Preprocessing**: Removed price outliers (top 1%) and standardized features using `StandardScaler` to ensure model stability.
3. **Feature Engineering**: 
    - Created `house_age`: The age of the house at the time of the data collection.
    - Created `years_since_renovation`: Calculated the effective age based on the last renovation date.
4. **Modeling**: Trained a **Random Forest Regressor** with 100 estimators and a max depth of 15.
5. **Deployment**: Deployed a Python-based web app using **Streamlit**, hosted on Streamlit Community Cloud.

## 📈 Model Performance
- **R² Score**: 0.8721 (Explains 87.2% of the price variance)
- **RMSE**: ~$100,868 (Average prediction error)

---

## 📁 Repository Structure
- `app.py`: The Streamlit web application script.
- `house_model.joblib`: The trained and compressed Random Forest model.
- `scaler.joblib`: The saved StandardScaler for normalizing user inputs.
- `requirements.txt`: List of Python dependencies required to run the project.
- `COM 572 Week 4 Practice Dataset.csv`: The raw dataset used for training.

## ⚙️ How to Run Locally
1. Clone this repository:
   ```
   git clone https://github.com/kelly7y176/house-price-prediction.git
   ```
2. Install dependencies:  
   ```
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:  
   ```
   streamlit run app.py
   ```
