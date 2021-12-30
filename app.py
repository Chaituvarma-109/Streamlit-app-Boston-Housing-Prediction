import streamlit as st
import pickle

from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Boston Housing Price Prediction", layout="wide")


def main():
    st.title("Boston Housing ML model using Linear Regression")

    input_cont = st.container()

    with input_cont:
        indus = st.number_input("Enter the value")
        rm = st.number_input("Enter the no. of rooms")
        tax = st.number_input("Enter the tax rate")
        ptratio = st.number_input("Enter the people teacher ratio")
        lstat = st.number_input("Enter the lstat")

        filename = 'model.pickle'
        model = pickle.load(open(filename, 'rb'))
        prediction = model.predict([[indus, rm, tax, ptratio, lstat]])

        # 6.91, 6.211, 233.0, 17.9, 7.44
        # 18.10,5.852,666.0,20.2,29.97
        st.info(f"Model prediction is: {round(prediction[0], 2)}")


if __name__ == "__main__":
    main()
