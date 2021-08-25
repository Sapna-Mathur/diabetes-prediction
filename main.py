from flask import Flask, request, render_template
import numpy as np
import pickle

# Intialize flask app
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        pregs = int(request.form['pregs'])
        glucose = int(request.form['glucose'])
        bp = int(request.form['bp'])
        skt = int(request.form['skt'])
        insulin = int(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = int(request.form['age'])

    except:
        error = "Invalid input.Please enter only numbers in the fields."
        return render_template('error.html', result=error)


    features = np.array([(pregs, glucose, bp, skt, insulin, bmi, dpf, age)])
    prediction = model.predict(features)

    if prediction[0] == 0:
        msg = "No need to fear. You have no dangerous symptoms of the diabetes."
        return render_template('result.html', result=msg)
    else:
        msg = "Sorry! You have chances of getting the diabetes. Please consult the doctor immediately."
        return render_template('result.html', result=msg)


if __name__ == "__main__":
    # app.run(debug=True)
    app.run(host="localhost", port=8000, debug=True)
