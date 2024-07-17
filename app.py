from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("model.pkl")

# Define routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    carat = float(request.form['carat'])
    color = request.form['color']
    clarity = request.form['clarity']

    # Encode the color and clarity inputs
    color_encoded = [1 if color == c else 0 for c in ['D', 'E', 'F', 'G', 'H', 'I', 'J']]
    clarity_encoded = [1 if clarity == c else 0 for c in ['IF', 'IL', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2']]

    # Create the feature vector by concatenating carat, color_encoded, and clarity_encoded
    feature_vector = [carat] + color_encoded + clarity_encoded

    # Convert the feature vector into a NumPy array for prediction
    feature_vector = np.array([feature_vector])

    # Make predictions using the trained model
    prediction = model.predict(feature_vector)

    # Format the prediction to display in dollars
    predicted_price = "{:,.2f}".format(prediction[0])

    # Render the prediction template with the result
    return render_template('index.html', prediction=predicted_price)

if __name__ == '__main__':
    app.run(debug=True)
