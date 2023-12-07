
from flask import Flask, request, Response
from joblib import load
import numpy as np

# Create a Flask web application
app = Flask(__name__)

# Load the pre-trained Logistic Regression model from the specified file
my_lr_model = load('model/Regress_log.joblib')

# Define a route for the web application
@app.route("/get_predictions", methods=['POST'])
def get_predictions():
    try:
        # Retrieve JSON data from the incoming POST request
        data = request.json
        
        # Extract the 'mydata' field from the JSON data
        user_sent_this_data = data.get('mydata')

        # Convert the user's data to a NumPy array and reshape it to match the model input format
        user_number = np.array(user_sent_this_data).reshape(1, -1)

        # Make predictions using the loaded model
        model_prediction = my_lr_model.predict(user_number)

        # Return the model prediction as a response
        return Response(str(model_prediction))
    except Exception as e:
        # If an exception occurs during the prediction process, return an error response
        return Response(str(e), status=500)

# Run the Flask application when this script is executed
if __name__ == '__main__':
    # Start the Flask web server with debugging turned off
    app.run(debug=False)

