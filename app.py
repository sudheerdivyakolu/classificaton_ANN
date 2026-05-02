import streamlit as st
import pandas as pd
import tensorflow as tf
import pickle

# Load model and preprocessing objects
model = tf.keras.models.load_model("model.h5")

with open("label_encoder_gender.pkl", "rb") as f:
    label_encoder = pickle.load(f)

with open("one_hot_encoder_geo.pkl", "rb") as f:
    onehot_encoder = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


st.title("Customer Churn Prediction")

# User Input
input_data = {
    "CreditScore": st.number_input(
        "Credit Score", min_value=300, max_value=850, value=600
    ),

    "Geography": st.selectbox(
        "Geography", ["France", "Spain", "Germany"]
    ),

    "Gender": st.selectbox(
        "Gender", ["Male", "Female"]
    ),

    "Age": st.number_input(
        "Age", min_value=18, max_value=100, value=30
    ),

    "Tenure": st.number_input(
        "Tenure", min_value=0, max_value=10, value=3
    ),

    "Balance": st.number_input(
        "Balance", min_value=0.0, value=10000.0
    ),

    "NumOfProducts": st.number_input(
        "Number of Products", min_value=1, max_value=4, value=1
    ),

    "HasCrCard": st.selectbox(
        "Has Credit Card", ["Yes", "No"]
    ),

    "IsActiveMember": st.selectbox(
        "Is Active Member", ["Yes", "No"]
    ),

    "EstimatedSalary": st.number_input(
        "Estimated Salary", min_value=0.0, value=50000.0
    )
}


# Convert input to dataframe
input_df = pd.DataFrame([input_data])


# Encode binary columns
input_df["HasCrCard"] = input_df["HasCrCard"].map({
    "Yes": 1,
    "No": 0
})

input_df["IsActiveMember"] = input_df["IsActiveMember"].map({
    "Yes": 1,
    "No": 0
})


# Encode gender
input_df["Gender"] = label_encoder.transform(
    input_df["Gender"]
)


# One-hot encode geography
geo_encoded = onehot_encoder.transform(
    input_df[["Geography"]]
).toarray()

geo_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder.get_feature_names_out(["Geography"])
)


# Remove original geography and merge encoded columns
input_df = pd.concat(
    [input_df.drop("Geography", axis=1), geo_df],
    axis=1
)


# Scale input
input_scaled = scaler.transform(input_df)


# Predict
prediction = model.predict(input_scaled)

prediction_proba = prediction[0][0]


# Display result
st.write(f"Predicted probability of churn: {prediction_proba:.2%}")


if prediction_proba > 0.5:
    st.error("Customer is likely to churn.")
else:
    st.success("Customer is likely to stay.")