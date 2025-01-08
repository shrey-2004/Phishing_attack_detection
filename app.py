# Import necessary libraries
#from flask import Flask, request, jsonify, render_template,url_for,request
from flask import Flask, request, render_template,request
import joblib  # To load the model
import pickle

# Initialize the Flask app
app = Flask(__name__)

#@app.route("/")
#def hello():
 # return "Hello World!"

#load the model
# Load the trained ML model
    #model = joblib.load('D:\\Shivaleela\\ML\\URL\\AIURL\\AIURL\\Src\\ML\\models\\phishing.pkl')
with open("C:\\Shreya\\AIURL\\AIURL\\Src\\ML\\models\\phishing.pkl", 'rb') as data:
    model = pickle.load(data)


# Define a route for predictions
@app.route('/')
def home():
	return render_template('home_new.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
    data = message
    #data = "www.dghjdgf.com/paypal.co.uk/cycgi-bin/webscrcmd=_home-customer&nav=1/loading.php" # Get data from the POST request
    #if not data:
     #   return ({"error": "No input data provided"}), 400

    # Extract features from the JSON request and reshape them if necessary
    #features = data.get('features')
    #if not features:
     #   return ({"error": "No 'features' key in input data"}), 400
    
    try:        
        # Convert the features to the proper format if necessary
        prediction = model.predict([data])
        print(prediction)
        #return ({"prediction": prediction[0]})
        return render_template('result_new.html',prediction=prediction[0])
    except Exception as e:
        return ({"error": str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True)