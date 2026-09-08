#  Fake News Detector

A machine learning web application that classifies news text as **REAL** or **FAKE**.  
Built end-to-end — from data cleaning and model training to a working Flask web app.

Machine Learning project, built as part of my journey from full-stack web developer toward becoming an **AI Engineer**.

---

##  Why I Built This

Before starting, I spent a lot of time researching **which project to build**. I didn't just want to copy a tutorial — I wanted a project that would:

- Teach me the **core ML concepts** (data cleaning, feature extraction, training, evaluation)
- Use **text data (NLP)**, since my long-term goal is to work with LLMs and RAG
- Be something I could turn into a **real full-stack web app** using the Flask and frontend skills I already had

After comparing several ideas (Titanic prediction, customer churn, sentiment analysis), I chose **Fake News Detection** because it combines NLP fundamentals with a practical, real-world use case — and it's a strong stepping stone toward Generative AI.

---

## 🧠 What It Does

You paste any news text into the app, and the model predicts whether it looks **REAL** or **FAKE**, based on patterns it learned during training (for example, clickbait/urgency wording tends to be fake, while neutral/attributed reporting tends to be real).

---

##  Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python |
| Data handling | Pandas |
| ML model | Scikit-learn (Logistic Regression) |
| Text → numbers | TF-IDF (TfidfVectorizer) |
| Backend | Flask |
| Frontend | HTML, CSS, JavaScript |

---

##  How I Built It — Step by Step

I built this project one step at a time, making sure I **understood each line** instead of just copy-pasting code.

### Step 1 — Getting the Data
Started with a CSV dataset of news articles, where each row had the news `text` and a `label` (REAL or FAKE).

### Step 2 — Exploring the Data
Loaded it with Pandas and checked the shape, the first few rows, and how many REAL vs FAKE examples there were.

### Step 3 — Cleaning the Data
This taught me a lot about real-world messy data:
- Removed rows with **missing** `text` or `label`
- Fixed inconsistent labels (`real`, `Fake`, `" FAKE"` → all standardized to `REAL` / `FAKE`)
- Stripped extra whitespace and normalized casing

### Step 4 — Preparing Features and Target
- **X (features)** = the news `text`
- **y (target)** = the `label` (REAL/FAKE)

Since ML models only understand numbers:
- Converted the **labels** to numbers using `LabelEncoder` (REAL → 1, FAKE → 0)
- Converted the **text** to numbers using `TfidfVectorizer` (TF-IDF)

I learned that X and Y each need their own conversion, using different tools.

### Step 5 — Splitting the Data
Split the data into **80% training** and **20% testing** using `train_test_split`, so I could check whether the model actually learned or just memorized.

### Step 6 — Training the Model
Trained a **Logistic Regression** model (a classification model — since the answer is a category, not a number). Under the hood, this uses **Gradient Descent** to gradually reduce error, which I studied separately to understand what `.fit()` really does.

### Step 7 — Evaluating the Model
Measured **accuracy** on the test set. I also learned that very high accuracy on clean, simple data doesn't always mean the model will perform the same on messy real-world data.

### Step 8 — Saving the Model
Saved three files with `joblib`:
- `model.pkl` — the trained model
- `vectorizer.pkl` — the TF-IDF vectorizer
- `label_encoder.pkl` — the label encoder

This separates **training** (done once) from **using** the model (done many times) — the model is trained once, saved, then loaded by the app.

### Step 9 — Building the Web App
- **Backend (Flask):** loads the three saved files and exposes a `/predict` API that takes news text and returns REAL/FAKE.
- **Frontend (HTML/CSS/JS):** a simple page with a text box and a button; JavaScript sends the text to the backend and displays the result.

---

## ▶️ How to Run

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the model** (creates the `.pkl` files)
   ```bash
   python train.py
   ```

3. **Run the app**
   ```bash
   python app.py
   ```

4. **Open in browser**
   ```
   http://127.0.0.1:5000
   ```

---

## 📁 Project Structure

```
fake-news-detector/
├── train.py              # trains and saves the model
├── app.py                # Flask backend
├── predict.py            # quick command-line testing (optional)
├── model.pkl             # saved model
├── vectorizer.pkl        # saved TF-IDF vectorizer
├── label_encoder.pkl     # saved label encoder
├── news_data.csv         # dataset
├── requirements.txt      # dependencies
├── README.md             # this file
├── .gitignore
└── templates/
    └── index.html        # frontend
```

---

## 🚀 What I Learned

- The **complete ML workflow**: data → cleaning → features → training → evaluation → deployment
- Why **text must be converted to numbers** (TF-IDF) — the foundation for embeddings and RAG later
- The difference between **classification and regression**, and why Logistic Regression fits this problem
- How to **save and load** a model so it powers a real application
- How to connect an ML model to a **Flask backend and a frontend** into one full-stack app

---

## 🔮 Next Steps

- Train on a larger, more realistic dataset to improve real-world accuracy
- Deploy the app online (Render / Railway)
- Continue toward my goal: **NLP → embeddings → LLMs → RAG → AI Agents**

---

*Built with curiosity, one step at a time. 🚀*
