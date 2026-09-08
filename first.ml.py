import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# 1. Data loadnews_data_big.csv
df = pd.read_csv('news_data_big.csv')

# print("checking")

# 2. Cleaning
df = df.dropna(subset=["text", "label"])
df["label"] = df["label"].str.strip()
df["label"] = df["label"].str.upper()
# print("checking")

# 3. X aur y
X = df["text"]
y = df["label"]

# print("checking")

# 4. label ko number banao (REAL/FAKE -> 1/0)
le = LabelEncoder()
y = le.fit_transform(y)


# 5. train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# print("checking")

# 6. X (text) ko numbers banao  ← YEH fit se PEHLE
vectorizer = TfidfVectorizer(stop_words="english")
X_train_num = vectorizer.fit_transform(X_train)
X_test_num = vectorizer.transform(X_test)

# print("checking")

# 7. model banao aur train karo
model = LogisticRegression()
model.fit(X_train_num, y_train)          # numbers do, text nahi

# print("checking")

# 8. predict aur accuracy
predictions = model.predict(X_test_num)  # predict mein X jaata hai, numbers wala
print("Accuracy:", accuracy_score(y_test, predictions))

# print("checking")

# apni khud ki news pe test karo
my_news = ["BREAKING: Government giving free laptops to everyone, register now before midnight!"]

# use bhi numbers mein badlo (wahi vectorizer)
my_news_num = vectorizer.transform(my_news)

# model se pucho
result = model.predict(my_news_num)

# result 0/1 hai, use wapas REAL/FAKE mein badlo
print("Result:", le.inverse_transform(result))


# print("checking")

joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(le, "label_encoder.pkl")

print("Teeno file save ho gayi!")
