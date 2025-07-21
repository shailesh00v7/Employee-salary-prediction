import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Employee Salary Prediction", layout="centered")
st.title("Employee Salary Prediction App")

# File uploader
uploaded_file = st.file_uploader("Upload the Adult Census CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("Raw Data Preview")
    st.write(data.head())

    # Data Cleaning
    data['workclass'].replace({'?': 'Others'}, inplace=True)
    data['occupation'].replace({'?': 'Others'}, inplace=True)
    data = data[~data['workclass'].isin(['Without-pay', 'Never-worked'])]
    data = data[(data['age'] <= 75) & (data['age'] >= 17)]
    data = data[(data['educational-num'] <= 16) & (data['educational-num'] >= 5)]
    data = data.drop(columns=['education'])

    # Save mapping for dropdowns
    cat_columns = data.select_dtypes(include='object').columns
    label_encoders = {}
    for col in cat_columns:
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(data[col])
        label_encoders[col] = encoder

    st.subheader("Cleaned Data Preview")
    st.write(data.head())

    if 'income' in data.columns:
        X = data.drop(columns=['income'])
        y = data['income']

        model_choice = st.selectbox("Choose a Model", ["Decision Tree", "Random Forest", "Logistic Regression"])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if st.button("Train Model"):
            if model_choice == "Decision Tree":
                model = DecisionTreeClassifier()
            elif model_choice == "Random Forest":
                model = RandomForestClassifier()
            else:
                model = LogisticRegression(max_iter=1000)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            st.session_state.trained_model = model
            st.session_state.columns = X.columns
            st.session_state.label_encoders = label_encoders

            st.subheader("Model Accuracy")
            st.success(f"Accuracy using {model_choice}: {acc * 100:.2f}%")

            if model_choice in ["Decision Tree", "Random Forest"] and st.checkbox("Show Feature Importances"):
                importances = model.feature_importances_
                feature_importance_df = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': importances
                }).sort_values(by='Importance', ascending=False)
                st.bar_chart(feature_importance_df.set_index('Feature'))

        st.subheader("Make a Prediction")

        if 'trained_model' in st.session_state:
            user_input = {}
            for col in st.session_state.columns:
                if col in st.session_state.label_encoders:
                    options = st.session_state.label_encoders[col].classes_.tolist()
                    selected = st.selectbox(f"{col}", options, key=col)
                    val = st.session_state.label_encoders[col].transform([selected])[0]
                else:
                    val = st.number_input(f"{col}", float(data[col].min()), float(data[col].max()), key=col)
                user_input[col] = val

            if st.button("Predict Income"):
                input_df = pd.DataFrame([user_input])
                prediction = st.session_state.trained_model.predict(input_df)
                st.success(f"Predicted Income Class: {'>50K' if prediction[0] == 1 else '<=50K'}")
        else:
            st.warning("Train the model first before making predictions.")

    else:
        st.error("The uploaded dataset must contain an 'income' column for prediction.")
else:
    st.info("Please upload a CSV file to begin.")
