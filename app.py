from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

model = pickle.load(open("insurance_model.pkl", "rb"))
columns = pickle.load(open("insurance_columns.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():

    input_dict = {
        "age": float(request.form["age"]),
        "bmi": float(request.form["bmi"]),
        "children": float(request.form["children"]),
        "sex_male": 1 if request.form["sex"] == "male" else 0,
        "smoker_yes": 1 if request.form["smoker"] == "yes" else 0,
        "region_northwest": 1 if request.form["region"] == "northwest" else 0,
        "region_southeast": 1 if request.form["region"] == "southeast" else 0,
        "region_southwest": 1 if request.form["region"] == "southwest" else 0
    }

    df = pd.DataFrame([input_dict])

    df = df.reindex(columns=columns, fill_value=0)

    prediction = model.predict(df)[0]

    return render_template("index.html",
                           prediction_text="Predicted Charges: ₹ " + str(round(prediction,2)))

if __name__ == "__main__":
    app.run(debug=True)