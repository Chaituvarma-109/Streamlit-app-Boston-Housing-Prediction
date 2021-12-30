import streamlit as st
import pickle

from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Boston Housing Price Prediction")


def main():
    st.title("Boston Housing ML model using Linear Regression")

    input_cont = st.container()

    with input_cont:
        indus = st.number_input("INDUS value")
        rm = st.number_input("RM value")
        tax = st.number_input("TAX rate")
        ptratio = st.number_input("PT RATIO Value")
        lstat = st.number_input("LSTAT Value")

    filename = 'model.pickle'
    model = pickle.load(open(filename, 'rb'))
    prediction = model.predict([[indus, rm, tax, ptratio, lstat]])

    # 6.91, 6.211, 233.0, 17.9, 7.44
    # 18.10,5.852,666.0,20.2,29.97
    st.subheader("Predicted Output")
    st.info(f"Model Predicted Price is: {round(prediction[0], 2)}")


if __name__ == "__main__":
    main()
