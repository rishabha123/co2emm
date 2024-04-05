from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

# Create Flask app
app = Flask(__name__)

# Load the pickle model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    try:
        # Convert form values to floats
        float_features = [float(request.form['engine_size']),
                          float(request.form['cylinders']),
                          float(request.form['fuel_consumption'])]

        # Reshape the input features for prediction
        features = [np.array(float_features).reshape(1, -1)]

        # Make prediction
        prediction = model.predict(np.array(float_features).reshape(1, -1))

        return render_template("index.html", prediction_text="The predicted CO2 emission is {}".format(prediction[0]))

    except ValueError as e:
        return render_template("index.html", prediction_text="Error: {}".format(str(e)))

if __name__ == "__main__":
    app.run(debug=True)
