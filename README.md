📊 Customer Churn Prediction App
Overview:

This project predicts the likelihood of telecommunication customers leaving the service (churn) using machine learning. The app provides actionable insights to help businesses retain customers by identifying high-risk customers early.

Live Demo:
    https://customer-churn-app-app-sdzs6cpc6hfcigx55a85dj.streamlit.app/

Project Structure:
    'scaler.pkl': The saved scaler for data preprocessing.
    'churn_model.pkl': The trained Gradient Boosting model.
    'app.py': The streamlit web application.

Features:
    Exploratory Data Analysis: Inisights into customer behavior
    Predict churn: Input customer details and receive a prediction along with probability.
    Feature importance: See the top factors influencing churn.
    Interactive uI: Built with Streamlit web app for real-time predictions, easy to use without coding. knowledge.

Dataset:
    Source: Telco Customer Churn Dataset
    Columns include customer info, services subscribed, contract type, monthly charges, and churn status.

Technologies:
    Python (pandas, numpy)
    Scikit-learn (Gradient Boosting Classifier)
    Streamlit (interactive web app)
    Joblib (saving/loading model and preprocessing objects)
    Matplotlib & Seaborn (visualizations)

Workflow:
    Data Cleaning & Preprocessing:
    Handled missing values, convert data types, one-hot encode categorical features, scale numeric features.
Model Training/Machine Learning:
    Trained Random Forest, XGBoost, and Gradient Boosting Classifiers.
    Gradient Boosting achieved the best performance with ~79% accuracy.
Saving Model:
    Saved trained model, scaler, and column order using joblib.
Deployment:
    Built a Streamlit app with numeric and categorical inputs.
    Predictions are aligned with training features and scaled correctly.
    Optional feature importance chart shows top 10 factors.
Installation:
    clone the repo:
    git clone <repo-url>
Install requirement:
    pip install -r requirements.txt
Run app:
    streamlit run app.py

Sample Prediction:
Input: Customer on a month-to-month contract, Fiber optic internet, monthly charge, tenure 12 months.
Output:
    Prediction churn: likely to churn
    Churn probability: 69.65%
Key Learnings:
    Feature scaling and alignment are crucial for live deployment.
    Gradient Boosting handles categorical and numeric features effectively.
    Streamlit allows rapid prototyping of interactive ML apps.

Contact:
Built by Idongesit Benson (comfort)
GitHub: github.com/comfortBenson