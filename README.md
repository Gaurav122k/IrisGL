# IrisGL
FLASK API CALL FOR PREDICTION OF LOGISTIC REGRESSION MODEL

1. Import Statements:
   - The code begins by importing necessary modules. Flask is used for creating the web application, and specific components like `request` and `Response` are imported for handling HTTP requests and responses. Additionally, the `joblib` module is imported to load a pre-trained machine learning model.
2. Flask App Initialization:
   - An instance of the Flask class is created with the variable name `app`. This instance represents the web application.
3. Loading the Machine Learning Model:
   - The code uses the `load` function from the `joblib` module to load a pre-trained machine learning model. The model is stored in the file 'Regress_log.joblib' in the 'model' directory. The loaded model is assigned to the variable `my_lr_model`.
4. API Endpoint Definition:
   - The code defines a single API endpoint '/get_predictions' using the `@app.route` decorator. This endpoint is configured to accept HTTP POST requests.
5. Request Processing:
   - Inside the '/get_predictions' endpoint, the code attempts to process incoming JSON data from the HTTP request. It expects a JSON object with a key 'mydata' containing an array of numerical data.
6. Model Prediction:
   - The numerical data sent by the user is converted into a NumPy array and reshaped to match the expected input shape of the machine learning model (`reshape(1, -1)`). The pre-trained model (`my_lr_model`) is then used to make predictions on this input data.
7. Response Handling:
   - The model's prediction is converted to a string and returned as the response to the client. In case of any exceptions during this process, a 500 Internal Server Error response is returned with details of the exception.
8. Application Execution:
   - The script ensures that the Flask app is only run when the script is executed directly (not imported as a module). The `app.run()` function starts the development server. The `debug` parameter is set to `False` to disable debugging mode in a production environment.
