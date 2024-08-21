from flask import Flask, request, render_template
import joblib

# Load the model components
clf = joblib.load('nationality_predictor_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def predict_nationality():
    if request.method == "POST":
        name = request.form["name"]
        prediction = predict([name], True)
        return render_template("index.html", prediction=prediction[0], name=name)
    return render_template("index.html")

def predict(names, label_str=False):
    name_vector = vectorizer.transform(names)
    pred = clf.predict(name_vector)
    if not label_str:
        return pred
    else:
        return label_encoder.inverse_transform(pred.ravel()).ravel()

if __name__ == "__main__":
    app.run(debug=True)
