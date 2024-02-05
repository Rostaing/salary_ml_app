import streamlit as st 
import pymongo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport
from streamlit_option_menu import option_menu
from streamlit_pandas_profiling import st_profile_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler, PolynomialFeatures
from lazypredict.Supervised import LazyRegressor
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import ExtraTreesRegressor
from yellowbrick.regressor import PredictionError
import time


client = pymongo.MongoClient("localhost", 27017)

st.set_page_config(page_title="Salary App | ML", page_icon="image/Logo_RostaingAI.jpeg")

html_temp = """
    <div style="background-color:#d33682;padding:15px;text-align:center;">
    <h2 style="color:white;">Machine Learning App</h2>
    </div>
"""

with st.sidebar:
    
    selected = option_menu(
        
        "Menu",
        ["DATASET", "EDA", "PREDICTION"],
        icons = ["database", "book", "buildings", "calendar2-check"],
        menu_icon = "cast",
        default_index = 0
    )

@st.cache_data
def get_data():
    
    
    db = client.test2
    employe = db.employes.find()
    # st.dataframe(employe)
    data = pd.DataFrame(employe)
    return data

def exploratory_data_analysis(x):
    
    profile = ProfileReport(x)
    st_profile_report(profile)
    
def prediction():
    
    data = get_data()
    data.dropna(axis=0, inplace=True)
    
    encoder = LabelEncoder()
    
    data["Age"] = data["Age"].apply(pd.to_numeric)
    data["Gender"] = encoder.fit_transform(data["Gender"])
    data["Education Level"] = encoder.fit_transform(data["Education Level"]).astype(int)
    data["Job Title"] = encoder.fit_transform(data["Job Title"]).astype(int)
    data["Years of Experience"] = data["Years of Experience"].apply(pd.to_numeric)
    data["Salary"] = data["Salary"].apply(pd.to_numeric)
    
    df = data.copy()
    df.drop("_id", axis=1, inplace=True)
    df.dropna(axis=0, inplace=True)
    
    for i in df.columns:
        df[i].clip(lower=df[i].quantile(0.05), upper=df[i].quantile(0.95), inplace=True)
    
    X = df[["Age", "Gender", "Education Level", "Job Title", "Years of Experience"]]
    y = df["Salary"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_train = y_train.values.ravel()  
    y_test = y_test.values.ravel()
    
    lpr = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
    models, predictions = lpr.fit(X_train, X_test, y_train, y_test)
     
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Dataset cleaned", "Data visualization processing", "Models", "Make prediction", "View model"])
    with tab1:
        st.write("", df)
        
    with tab2:
        st.write("", exploratory_data_analysis(df))
        
    with tab3:
        st.write("", models)
              
    with tab4:
        model = make_pipeline(PolynomialFeatures(degree=2), StandardScaler(), ExtraTreesRegressor(n_estimators=100))
        model.fit(X_train, y_train)
        st.write("", f"**Train score: {round(model.score(X_train, y_train), 2)*100}% | Test score: {round(model.score(X_test, y_test), 2)*100}%**")
        
        col1, col2, col3 = st.columns(3)
        
        dt = get_data()
        dt.dropna(axis=0, inplace=True)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=65)
            
        with col2:
            var_gender = dt["Gender"].unique()
            encoder.fit_transform(var_gender)
            gender = st.selectbox("Gender", var_gender)        
            selected_gender = encoder.transform([gender])[0]
            if selected_gender:
                gender = selected_gender
            else:
                st.warning("Please select a gender.")
                        
        with col3:
            var_education_level = dt["Education Level"].unique()
            encoder.fit_transform(var_education_level)
            education = st.selectbox("Education Level", var_education_level)
            selected_education = encoder.transform([education])[0]
            if selected_education:
                education = selected_education
            else:
                st.warning("Please select a Education Level.")
                      
        with col1:
            var_job_title = dt["Job Title"].unique()
            encoder.fit_transform(var_job_title)
            job_title =  st.selectbox("Job Title", var_job_title)
            selected_job_title = encoder.transform([job_title])[0]
            if selected_education:
                job_title = selected_job_title
            else:
                st.warning("Please select a Job Title.")
            
        with col2:
            year_experience = st.number_input("Year of Experience", min_value=0, max_value=50)
        
        y_pred = model.predict([[age, gender, education, job_title, year_experience]]).flatten()[0]
        
        if st.button("Prediction"):
            with st.spinner("In progress..."):
                time.sleep(3)
                st.write(f"Le Salaire de cet employé s'élève à {round(y_pred, 2)} € par an et à {round(y_pred / 12, 2)} € par mois.")
                st.balloons()
                
        st.divider()
                
    with tab5:
        fig = plt.figure(figsize=(4, 6))
        viz = PredictionError(model)
        viz.fit(X_train, y_train)
        viz.score(X_test, y_test)
        viz.show()
        st.pyplot(fig)
            

def main():
    
    if selected == "DATASET":
        st.markdown(html_temp, unsafe_allow_html=True)
        st.title("Unprocessed dataset")
        st.balloons()
        st.dataframe(get_data())
        
    elif selected == "EDA":
         st.balloons()
         exploratory_data_analysis(get_data())
         
    elif selected == "PREDICTION":
        prediction()
        
if __name__ == "__main__":
    main()
    
    
