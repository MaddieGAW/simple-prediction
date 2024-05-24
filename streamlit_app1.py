import streamlit as st
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Page configuration
st.set_page_config(
    page_title='Predicting the Success of International Development Aid Projects',
    page_icon='🌷',
    layout='wide',
    initial_sidebar_state='expanded')

# Title of the app
st.title('🌷 Project Success Calculator')

# Load Data
cleaned_df = pd.read_csv('output.csv')

# Preprocessing for numerical data
scaler = StandardScaler()

def load_model():
    # Define feature and target variables
    X = cleaned_df.drop('success', axis=1)
    y = cleaned_df['success']

    # Preprocessing for numerical and categorical data
    numerical_features = ['project_duration', 'eval_lag', 'sector_code', 'completion_year', 'project_size_USD_calculated']

    # Create the column transformer
    preprocessor = ColumnTransformer(
        transformers=[('num', StandardScaler(), numerical_features)])

    # Create a pipeline that includes the preprocessor and the classifier
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=26)

    # Train the model
    clf.fit(X_train, y_train)

    return clf

clf = load_model()

# Preprocessing for numerical and categorical data
numerical_features = ['project_duration', 'eval_lag', 'sector_code', 'completion_year', 'project_size_USD_calculated']

# Define function to preprocess input data
def preprocess_input(data):
    # Preprocess numerical features
    data[numerical_features] = scaler.transform(data[numerical_features])
    return data

# Function to make prediction
def predict(project_duration, eval_lag, sector_code, completion_year, project_size_USD_calculated):
    input_data = pd.DataFrame({
        'project_duration': [project_duration],
        'eval_lag': [eval_lag],
        'sector_code': [sector_code],
        'completion_year': [completion_year],
        'project_size_USD_calculated': [project_size_USD_calculated]
    })
    input_data = preprocess_input(input_data)
    prediction = clf.predict(input_data)
    return prediction[0]

# Streamlit UI
st.title('Project Prediction App')

project_duration = st.slider('Project Duration', min_value=0, max_value=1000, step=1)
eval_lag = st.slider('Evaluation Lag', min_value=0, max_value=1000, step=1)
sector_code = st.selectbox('Sector Code', options=[1, 2, 3])  # Example options, replace with your actual options
completion_year = st.slider('Completion Year', min_value=2000, max_value=2030, step=1)
project_size_USD_calculated = st.number_input('Project Size (USD)', value=1000)

if st.button('Predict'):
    prediction = predict(project_duration, eval_lag, sector_code, completion_year, project_size_USD_calculated)
    st.write('Predicted Output:', prediction)