import streamlit as st 
import pickle 

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

# Function to predict CO2 consumption
def predict(size, cylinder, consumption):
    result = model.predict([[size, cylinder, consumption]])
    return result[0]  # Return the predicted value

def main():
    st.title('CO2 Consumption Model')
    html_template = """<div style="background-color: lightblue; padding: 20px; border-radius: 10px;"><h1>CO2 Emission Model</h1></div>"""
    st.markdown(html_template, unsafe_allow_html=True)
    
    # Input fields for user input
    size = st.text_input("Engine Size", "Type Here")
    cylinder = st.text_input("Cylinders", "No of Cylinders")
    consumption = st.text_input("Fuel Consumption", "Type Here")
    
    result = ""
    if st.button("Predict"):
        # Convert input values to numeric types
        try:
            size = float(size)
            cylinder = float(cylinder)
            consumption = float(consumption)
            result = predict(size, cylinder, consumption)
        except ValueError:
            st.error("Please enter valid numeric values for engine size, cylinders, and fuel consumption.")

    st.success('The CO2 consumption is {}'.format(result))

    if st.button("Info"):
        st.text("This is an ML model designed to predict CO2 Emission using Decision Tree Classifier, prepared with the help of Streamlit")

if __name__ == "__main__":
    main()
