from flask import Flask, render_template, request
import numpy as np
import pickle

diabetes_model = pickle.load(open('modeldd.pkl', 'rb'))


app = Flask(__name__)


@app.route("/diabetes", methods=['GET','POST'])
def diabetes():
    return render_template('diabetes.html')


@app.route("/predict", methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        if(len([float(x) for x in request.form.values()])==8):
            preg = int(request.form['pregnancies'])
            glucose = int(request.form['glucose'])
            bp = int(request.form['bloodpressure'])
            st = int(request.form['skinthickness'])
            insulin = int(request.form['insulin'])
            bmi = float(request.form['bmi'])
            dpf = float(request.form['dpf'])
            age = int(request.form['age'])
            
            data = np.array([[preg,glucose, bp, st, insulin, bmi, dpf, age]])
            my_prediction = diabetes_model.predict(data)
            
            return render_template('diabetes.html', prediction_text='Diabetes Prediction: {}'.format(my_prediction))







if __name__ == "__main__":
    app.run(debug=True)