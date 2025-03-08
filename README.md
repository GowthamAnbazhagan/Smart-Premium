💰 SmartPremium: Predicting Insurance Costs with Machine Learning
📌 Project Overview
Insurance companies estimate premiums based on multiple factors like age, income, health status, and claim history. This project develops a machine learning model to accurately predict insurance premium amounts using Linear Regression, Decision Trees, Random Forest, and XGBoost.

The project follows a structured ML pipeline, includes MLflow for experiment tracking, and is deployed using Streamlit for real-time premium predictions.

🚀 Key Features
✅ Data Preprocessing & Feature Engineering (Handling missing values, encoding categorical data)
✅ Exploratory Data Analysis (EDA) (Visualizing relationships between features)
✅ Regression Modeling & Evaluation (Training models & selecting the best one)
✅ Hyperparameter Tuning (Optimizing model performance)
✅ ML Pipeline & MLflow Integration (Automating workflow & tracking experiments)
✅ Streamlit Deployment (User-friendly web app for real-time predictions)

🏢 Business Use Cases
🔹 Insurance Companies: Optimize premium pricing based on risk factors.
🔹 Financial Institutions: Assess risk for loans tied to insurance policies.
🔹 Healthcare Providers: Estimate future healthcare costs for patients.
🔹 Customer Service: Provide real-time insurance quotes using ML.

📊 Dataset Overview
This dataset contains 200,000+ records with 20 features, including:

Age, Income, Health Score, Occupation, Smoking Status, Credit Score, Previous Claims, Policy Type, Premium Amount (Target variable)
Features include numerical, categorical, and text data, with missing values and skewed distributions, mimicking real-world datasets.
🔍 Approach
📌 Step 1: Data Understanding & Exploration
Load the dataset & perform EDA using Pandas, Matplotlib, and Seaborn.
Identify missing values, skewed distributions, and correlations.
📌 Step 2: Data Preprocessing
Handle missing values (median/mode imputation).
Convert categorical variables (One-Hot Encoding, Label Encoding).
Feature Scaling (StandardScaler, MinMaxScaler).
📌 Step 3: Model Development
Train multiple regression models:
✅ Linear Regression
✅ Decision Trees
✅ Random Forest
✅ XGBoost
Evaluate using R² Score, RMSE, MAE, RMSLE to select the best model.
📌 Step 4: ML Pipeline & MLflow Integration
Automate the training process using ML pipelines.
Track model performance, parameters, and results using MLflow.
📌 Step 5: Model Deployment with Streamlit
Build an interactive web app for real-time premium predictions.
Users can input customer details and receive instant premium estimates.
📈 Project Evaluation Metrics
✔ Root Mean Squared Logarithmic Error (RMSLE)
✔ Root Mean Squared Error (RMSE)
✔ Mean Absolute Error (MAE)
✔ R² Score

🛠 Tech Stack
🔹 Python (Pandas, NumPy, Scikit-Learn, XGBoost)
🔹 MLflow (Experiment tracking)
🔹 Streamlit (Web app deployment)
🔹 Git/GitHub (Version control)
🔹 Matplotlib, Seaborn (EDA & Visualizations)

🎯 Results
📊 Achieved low error rates for premium predictions.
✅ Developed a fully functional Streamlit web app for real-time insurance cost estimation.


📂 Project Deliverables
📁 Jupyter Notebook (Full ML pipeline)
📁 MLflow experiment tracking setup
📁 Trained Model for Deployment
📁 Streamlit Web App

📜 License
This project is for educational purposes and can be freely used for practice.

📧 Contact
📌 Created by: Gowtham Anbazhagan
📧 Email: gowthamanbazhagan@gmail.com
🔗 LinkedIn: Gowtham Anbazhagan
