import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pandas as pd
import pickle

#Load the trained model
model = tf.keras.models.load_model('model.h5')

#Load the encoders and scaler files
with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)

with open('label_encoder_ggeo.pkl','rb') as file:
    label_encoder_ggeo = pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)


## Streamlit App
st.title('Customer Churn Prediction')

#User Input
geography = st.selectbox('Geography', label_encoder_ggeo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age',18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0,10)
num_of_product = st.slider('NUmber of Products',1,4)
has_cr_card = st.selectbox('Has Credit Card',[0,1])
is_active_member = st.selectbox('Is Active Member',[0,1])

#Prepare the input data
input_data = pd.DataFrame({
'CreditScore':[credit_score],
'Gender':[label_encoder_gender.transform([gender])[0]],
'Age':[age],
'Tenure':[tenure],
'Balance':[balance],
'NumOfProducts':[num_of_product],
'HasCrCard':[has_cr_card],
'IsActiveMember':[is_active_member],
'EstimatedSalary':[estimated_salary]
})


## One Hot Encoding for Geograohy COlumns
geo_encoded = label_encoder_ggeo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded , columns = label_encoder_ggeo.get_feature_names_out(['Geography']))

#Combine one hot encoded columns with input data
input_df = pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis = 1)

#Sacle the entire data
input_data_scaled = scaler.transform(input_df)

#Predict Churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5 :
    print('The Customer is likely to chur')
else:
    print('The Customer is not likely to chur')







