import streamlit as st
import joblib

# Load the saved model and vectorizer
model = joblib.load("spam_detection_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Streamlit app layout
st.title("Spam Mail Detection")
st.markdown("""
This app uses a Machine Learning model to detect whether an email is spam or not.  
Enter the text of the email below to get the prediction.
""")

# Input field for email text
email_text = st.text_area("Enter the email content:", height=200)

# Prediction button
if st.button("Predict"):
    if email_text.strip() == "":
        st.warning("Please enter some email content.")
    else:
        # Transform the input text using the loaded vectorizer
        email_tfidf = vectorizer.transform([email_text])
        
        # Predict using the loaded model
        prediction = model.predict(email_tfidf)
        
        # Display the result
        if prediction[0] == 1:
            st.error("This email is classified as **Spam**.")
        else:
            st.success("This email is classified as **Ham** (Not Spam).")
