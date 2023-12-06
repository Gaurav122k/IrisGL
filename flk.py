
from flask import Flask, request, Response
from joblib import load
import numpy as np

app = Flask(__name__)

my_lr_model = load('model/Regress_log.joblib')

@app.route("/get_predictions", methods=['POST'])
def get_predictions():
    try:
        data = request.json
        user_sent_this_data = data.get('mydata')

        user_number = np.array(user_sent_this_data).reshape(1, -1)

        model_prediction = my_lr_model.predict(user_number)

        return Response(str(model_prediction))
    except Exception as e:
        return Response(str(e), status=500)

if __name__ == '__main__':
    app.run(debug=False)
