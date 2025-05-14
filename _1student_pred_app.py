import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler,LabelEncoder

#loading pickle file
def load_model():
    with open("_1student_lr_final_model.pkl","rb") as file:
        model,scaler,le = pickle.load(file)
    return model,scaler,le

#preprocessing output
def preprocessing_input_data(data,scaler,le):
    data['Extracurricular Activities'] = le.transform([data["Extracurricular Activities"]])[0]
    df = pd.DataFrame([data])
    df_transformed = scaler.transform(df)
    return df_transformed

#predict 
def predict_data(data):
    model,scaler,le = load_model()
    processed_data = preprocessing_input_data(data,scaler,le)
    prediction = model.predict(processed_data)
    return prediction

#creating UI:
def main():
 
    st.title("student performance prediction")
    st.write("enter your data to get a prediction for your performance")

    hour_studied = st.number_input("Hours Studied",min_value=1,max_value=10,value=6)
    prvious_score = st.number_input("Previous Scores",min_value=40,max_value=100,value=60)
    extra = st.selectbox("Extracurricular Activities",["Yes","No"])
    sleeping_hour = st.number_input("Sleep Hours",min_value=1,max_value=100,value=6)
    number_of_peper_solved = st.number_input("Sample Question Papers Practiced",min_value=1,max_value=100,value=6)

    if st.button("predict-your_score"):
        user_data = {
            "Hours Studied" : hour_studied,
            "Previous Scores" : prvious_score,
            "Extracurricular Activities" : extra,
            "Sleep Hours" : sleeping_hour,
            "Sample Question Papers Practiced" : number_of_peper_solved
        }    
        prediction = predict_data(user_data)
        st.success(f"your prediction result is {prediction}")

if __name__ == "__main__":
    main()
