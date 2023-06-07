#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report


# In[ ]:


# Load the dataset
dataset = pd.read_csv("Covid19_Symptoms.csv")


# In[ ]:


# Remove unnecessary columns
unused = ['Gender_Female', 'Gender_Male', 'Gender_Transgender', 'Contact_Dont-Know', 'Contact_No', 'Contact_Yes', 'Country', 'None_Sympton']
dataset = dataset.drop(columns=unused)


# In[ ]:


# Preprocess the dataset
severity_columns = dataset.filter(like='Severity_').columns
dataset['Severity_None'].replace({1:'None',0:'No'}, inplace=True)
dataset['Severity_Mild'].replace({1:'Mild',0:'No'}, inplace=True)
dataset['Severity_Moderate'].replace({1:'Moderate',0:'No'}, inplace=True)
dataset['Severity_Severe'].replace({1:'Severe',0:'No'}, inplace=True)
dataset['Severity'] = dataset[severity_columns].values.tolist()


# In[ ]:


def removing(listSeverity):
    listSeverity = set(listSeverity) 
    listSeverity.discard("No")
    newListSeverity = ''.join(listSeverity)
    return newListSeverity


# In[ ]:


dataset['Severity'] = dataset['Severity'].apply(removing)


# In[ ]:


age_columns = dataset.filter(like='Age_').columns
dataset['Age_0-9'].replace({1:'0-9',0:'No'}, inplace=True)
dataset['Age_10-19'].replace({1:'10-19',0:'No'}, inplace=True)
dataset['Age_20-24'].replace({1:'20-24',0:'No'}, inplace=True)
dataset['Age_25-59'].replace({1:'25-59',0:'No'}, inplace=True)
dataset['Age_60+'].replace({1:'60+',0:'No'}, inplace=True)
dataset['Age'] = dataset[age_columns].values.tolist()


# In[ ]:


def removing(listAge):
    listAge = set(listAge) 
    listAge.discard("No")
    newListAge = ''.join(listAge)
    return newListAge


# In[ ]:


dataset['Age'] = dataset['Age'].apply(removing)


# In[ ]:


duplicates = ['Age_0-9','Age_10-19', 'Age_20-24', 'Age_25-59', 'Age_60+', 'Severity_Mild', 'Severity_Moderate', 'Severity_None', 'Severity_Severe']
dataset = dataset.drop(columns=duplicates)


# In[ ]:


age_mapping = {
    "0-9": 1,
    "10-19": 2,
    "20-24": 3,
    "25-59": 4,
    "60+": 5
}
dataset['Age'] = dataset['Age'].map(age_mapping)


# In[ ]:


# Split the dataset into features and target
X = dataset.drop(columns='Severity')
y = dataset['Severity']


# In[ ]:


# Train the decision tree classifier
tree = DecisionTreeClassifier(max_depth=3, criterion="entropy", min_samples_leaf=2, random_state=0)
tree.fit(X, y)


# In[ ]:


# Streamlit app
st.title("COVID-19 Severity Classification")


# In[ ]:


# Display symptoms checkboxes
st.sidebar.title("Symptoms")
symptoms = ['Fever', 'Tiredness', 'Dry-Cough', 'Difficulty-in-Breathing', 'Sore-Throat', 'Pains', 'Nasal-Congestion', 'Runny-Nose', 'Diarrhea']
selected_symptoms = st.sidebar.multiselect("Select symptoms:", symptoms)


# In[ ]:


# Prepare user input for prediction
input_data = {symptom: 1 if symptom in selected_symptoms else 0 for symptom in symptoms}
input_df = pd.DataFrame([input_data])


# In[ ]:


# Make predictions
prediction = tree.predict(input_df)


# In[ ]:


# Display the predicted severity
st.subheader("Predicted Severity")
st.write(prediction[0])


# In[ ]:


# Show classification report
st.subheader("Classification Report")
y_pred = tree.predict(X)
report = classification_report(y, y_pred)
st.text_area("Classification Report:", report)

