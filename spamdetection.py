import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st
import time

# Read and preprocess the dataset
data = pd.read_csv("C:/Users/amits/OneDrive/Desktop/email spam detection/spam (1).csv")
data.drop_duplicates(inplace=True)
data['Category'] = data['Category'].replace(['ham', 'spam'], ['not spam', 'spam'])

mess = data['Message']
cat = data['Category']

# Train-test split
mess_train, mess_test, cat_train, cat_test = train_test_split(mess, cat, test_size=0.2)

# Vectorization (convert text to features)
cv = CountVectorizer(stop_words='english')
features = cv.fit_transform(mess_train)

# Naive Bayes model training
model = MultinomialNB()
model.fit(features, cat_train)

# Function to predict spam/ham
def predict(message):
    input_message = cv.transform([message]).toarray()
    result = model.predict(input_message)
    return result[0]

# Streamlit UI Styling
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to right, #f9f9f9, #e0f7fa);
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        height: 100vh;
        font-family: 'Poppins', sans-serif;
    }
    h1 {
        text-align: center;
        color: #333333;
        font-size: 3rem;
        font-weight: bold;
        margin-top: 2rem;
    }
   
    .stButton button {
        background: linear-gradient(135deg, #89f7fe, #66a6ff);
        color: white;
        border-radius: 30px;
        padding: 10px 25px;
        font-size: 1.2rem;
        margin-top: 1rem;
        font-weight: bold;
        border: none;
        box-shadow: 0px 4px 20px rgba(0,0,0,0.2);
        transition: all 0.3s ease-in-out;
    }
    .stButton button:hover {
        background: linear-gradient(135deg, #66a6ff, #89f7fe);
        transform: scale(1.05);
        cursor: pointer;
    }
    .stWrite {
        color: #444444;
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        margin-top: 2rem;
    }
    .stImage img {
        border-radius: 20px;
        margin-top: 2rem;
        box-shadow: 0px 8px 25px rgba(0,0,0,0.15);
        transition: transform 0.3s ease;
    }
    .stImage img:hover {
        transform: scale(1.03);
    }
    </style>
    """, unsafe_allow_html=True
)

# --- Main App ---

# Page Title
st.markdown("<h1>📧 Email Spam Detector </h1>", unsafe_allow_html=True)

# Create a form for input and button together
with st.form(key="spam_form", clear_on_submit=True):
    input_mess = st.text_input('✉️ Type your message here')
    submit_button = st.form_submit_button(label='🔍 Check Now')

# Prediction after submission
if submit_button:
    if input_mess.strip() != "":
        with st.spinner('🔍 Checking your message...'):
            time.sleep(1)  # optional for effect

            output = predict(input_mess)

            if output == "spam":
                st.image("a11.jpeg", width=400)
                st.write("🚨 **Warning: This is a spam message!**")
            else:
                st.image("a222.png", caption="✅ Safe message", width=400)
                st.write("✅ **This is a safe message!**")
                st.balloons()
