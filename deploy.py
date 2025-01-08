# Import necessary libraries
from Flask import Flask, request, jsonify
import joblib  # To load the model

# Initialize the Flask app
app = Flask(__name__)

# Load the trained ML model
model = joblib.load('C:\Shreya\Mini project_2\AI-Based-Threat-Intelligence-and-Prediction-System-main\AI-Based-Threat-Intelligence-and-Prediction-System-main\ML\phishing.pkl')

# Define a route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Get the JSON data from request
    prediction = model.predict([data['input']])  # Make prediction
    return jsonify({'prediction': prediction[0]})  # Return as JSON

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
