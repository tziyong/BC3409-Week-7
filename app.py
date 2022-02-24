from flask import Flask
from flask import request, render_template
from scipy import stats

import pandas as pd
import joblib

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        income = float(request.form.get("income"))
        age = float(request.form.get("age"))
        loan = float(request.form.get("loan"))

        # Normalize input
        df = pd.read_csv('Credit Card Default II (balance).csv')
        df = df.append({'income': income, 'age': age, 'loan': loan}, ignore_index=True)
        X = df.loc[:, ["income", "age", "loan"]]
        Y = df.loc[:, "default"]

        # Normalization
        for i in X.columns:
            X[i] = stats.zscore(X[i].astype(float))

        # Get input from dataframe
        inputs = list(X.iloc[-1])
        income = inputs[0]
        age = inputs[1]
        loan = inputs[2]
        print(income, age, loan)

        model = joblib.load("cc_default_best_model")
        pred = model.predict([[income, age, loan]])
        pred = pred[0]
        print(pred)

        if pred == 0:
            result = 'No default risk'
        else:
            result = 'There is default risk'
        s = f"The predicted credit card default rate is: {result}"
        return render_template("index.html", result=s)
    else:
        return render_template("index.html", result="Predict 2")


if __name__ == "__main__":
    app.run()
