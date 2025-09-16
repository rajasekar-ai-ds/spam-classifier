
import streamlit as st
import joblib
from pathlib import Path
Base = Path(__file__).resolve().parent
model_path = Base / "spam_classifier.joblib"
pipeline = joblib.load(model_path)


st.title("📨AI Email Spam Classifier")
st.caption("By~ Rajasekar , Saravanan , Raghav")

option = st.radio("Choose an option to continue: ", ["📝Enter a Message", "📂Txt File upload"] ,index=None)
if option:
    if option == "📝Enter a Message":

        message = st.text_input("✍Enter a Message")

        if st.button("Predict"):

            if not message or message.strip() == "":
                st.warning("Please enter a message: ")
            else:
                with st.spinner("Predicting......"):
                    result = pipeline.predict([message])
                    proba = pipeline.predict_proba([message]) [0] if hasattr(pipeline , 'predict_proba') else None
                    spam_conf = None
                    if proba is not None:
                        spam_conf = proba[1] if len(proba) > 1 else proba[0]
                        not_spam_conf = 1 - spam_conf
                    if result == 1:
                        if spam_conf is not None:
                            st.markdown(f"<div style='padding:12px; border-radius:8px;background:#FF0000;color:white'>"
                                        f"  <strong> 🚨SPAM </strong> -{spam_conf*100:.1f}% confident </div>" ,
                                        unsafe_allow_html=True)
                        else: st.error("🚨SPAM")

                    else:
                        if spam_conf is not None:
                            st.markdown(f"<div style='padding:12px; border-radius:8px; background:#2ecc71; color:white'>"
                                        f"  <strong> ✅NOT SPAM </strong> -{not_spam_conf*100:.1f}% confident </div>",
                                        unsafe_allow_html=True)
                        else:
                            st.success("✅NOT SPAM")
    elif option == "📂Txt File upload":
        upload = st.file_uploader("Upload a txt file : ", type=['txt'])
        if upload is not None:
            text = upload.read().decode("utf-8", errors="ignore")
            if st.button("Predict"):
                result = pipeline.predict([text])
                proba = pipeline.predict_proba([text])[0] if hasattr(pipeline, 'predict_proba') else None
                spam_conf = None
                if proba is not None:
                    spam_conf = proba[1] if len(proba) > 1 else proba[0]
                    not_spam_conf = 1 - spam_conf
                if result == 1:
                    if spam_conf is not None:
                        st.markdown(f"<div style='padding:12px; border-radius:8px;background:#ff3d4d;color:white'>"
                                    f"  <strong> SPAM </strong> -{spam_conf * 100:.1f}% confident </div>",
                                    unsafe_allow_html=True)
                    else:
                        st.error("SPAM")

                else:
                    if spam_conf is not None:
                        st.markdown(f"<div style='padding:12px; border-radius:8px; background:#2ecc71; color:white'>"
                                    f"  <strong> NOT SPAM </strong> -{not_spam_conf * 100:.1f}% confident </div>",
                                    unsafe_allow_html=True)
                    else:
                        st.success("NOT SPAM")








