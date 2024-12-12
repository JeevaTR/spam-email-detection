import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Title of the Streamlit app
st.title("Spam Classifier")

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("spam.csv", encoding='latin-1')
    df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
    df.rename(columns={'v1': 'label', 'v2': 'message'}, inplace=True)
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    return df

df = load_data()

# Display dataset
st.write("spam.csv")
st.write(df.head())

# Train model
@st.cache_data
def train_model(df):
    cv = CountVectorizer()
    X = cv.fit_transform(df['message'])
    X_train, X_test, y_train, y_test = train_test_split(X, df['label'], test_size=0.2, random_state=42)
    mnb = MultinomialNB()
    mnb.fit(X_train, y_train)
    accuracy = mnb.score(X_test, y_test)
    y_pred = mnb.predict(X_test)
    return mnb, cv, accuracy, y_test, y_pred

mnb, cv, accuracy, y_test, y_pred = train_model(df)

# Ensure y_test and y_pred are binary
y_test = y_test.astype(int)
y_pred = y_pred.astype(int)

# Display model accuracy
st.write(f"### Model Accuracy: {accuracy}")

# Debug prints
st.write("### Debug Info")
st.write(f"y_test unique values: {pd.Series(y_test).unique()}")
st.write(f"y_pred unique values: {pd.Series(y_pred).unique()}")

# Display confusion matrix
st.write("### Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', ax=ax)
st.pyplot(fig)

# Display classification report
st.write("### Classification Report")
st.text(classification_report(y_test, y_pred))

# Live predictor
st.write("### Live Predictor")
user_input = st.text_input("Enter a message to classify")

if user_input:
    user_message_transformed = cv.transform([user_input])
    prediction = mnb.predict(user_message_transformed)
    st.write(f'This is a {"spam" if prediction[0] == 1 else "ham"} message')
