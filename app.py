import streamlit as st
import pickle
import numpy as np
import re

rf = pickle.load(open("models/rf_model.pkl", "rb"))
w2v = pickle.load(open("models/w2v_model.pkl", "rb"))

def preprocess(text):
    text = re.sub(r"[^a-z A-Z ]", " ", text)
    return text.lower().split()

def avg_word2vec(tokens):
    vectors = [w2v.wv[word] for word in tokens if word in w2v.wv]
    if len(vectors) == 0:
        return np.zeros(w2v.vector_size)
    return np.mean(vectors, axis=0)

st.set_page_config(page_title="Spam Detector", page_icon="📧")

st.title("📧 Spam Detection System")
# st.write("Spam or Ham classification using Average Word2Vec + Random Forest")

message = st.text_area(
    "Enter email or SMS text",
    height=150,
)

if st.button("Predict"):
    if message.strip() == "":
        st.warning("⚠️ Please enter a message")
    else:
        tokens = preprocess(message)

        if len(tokens) < 2:
            st.info("Message too short to classify")
        else:
            vec = avg_word2vec(tokens).reshape(1, -1)

            prediction = rf.predict(vec)[0]
            proba = rf.predict_proba(vec)[0]

            if prediction == 1:
                st.error(f"🚨 Spam Message ")
                st.error(f" Spam probability : {proba[1]*100:.2f}% ")
            else:
                st.success(f"✅ Ham Message ")
                st.success(f" Ham probability : {proba[0]*100:.2f}% ")

st.markdown("---")


