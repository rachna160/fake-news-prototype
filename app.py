import streamlit as st
import joblib
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="AI Fake News Detector", page_icon="📰", layout="centered")

# Title
st.markdown("<h1 style='text-align:center; color:#2E86C1;'>📰 AI Fake News Detector</h1>", unsafe_allow_html=True)

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Text input
st.write("### Enter any news headline or paragraph:")
user_text = st.text_area("Type the news content here...", height=150)
if st.button("Analyze News"):
    if user_text.strip() == "":
        st.warning("⚠ Please enter some text!")
    else:
        x = vectorizer.transform([user_text])
        prediction = model.predict(x)[0]

        if prediction == "fake":
            st.markdown("<h2 style='color:red;'>❌ Fake News Detected!</h2>", unsafe_allow_html=True)
            st.image("https://cdn-icons-png.flaticon.com/512/4637/4637580.png", width=150)
        else:
            st.markdown("<h2 style='color:green;'>✅ Real News Detected!</h2>", unsafe_allow_html=True)
            st.image("https://cdn-icons-png.flaticon.com/512/148/148767.png", width=150)

# Footer
st.markdown("---")
st.markdown("<p style='text-align:center;'>Team <b>Innovibe</b> | Project by <b>Rachna Patel</b></p>", unsafe_allow_html=True)