from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# teeno load — shuru mein ek baar
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
le = joblib.load("label_encoder.pkl")

# check karne wala function
def check_news(text):
    text_num = vectorizer.transform([text])
    result = model.predict(text_num)
    return le.inverse_transform(result)[0]

# home page dikhao
@app.route("/")
def home():
    return render_template("index.html")

# predict API
@app.route("/predict", methods=["POST"])
def predict():
    news = request.json["text"]
    answer = check_news(news)
    return jsonify({"result": answer})

if __name__ == "__main__":
    app.run(debug=True)



    