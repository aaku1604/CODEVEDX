import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# -----------------------------
# Load Data
# -----------------------------
def load_data():
    try:
        return pd.read_csv("news.csv")
    except:
        print("Error loading dataset!")
        return pd.DataFrame(columns=["text", "label"])

# -----------------------------
# Train Model
# -----------------------------
def train_model(data):
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(data["text"])
    y = data["label"]

    model = MultinomialNB()
    model.fit(X, y)

    return model, vectorizer

# -----------------------------
# Predict News
# -----------------------------
def predict_news(model, vectorizer):
    text = input("Enter news text: ")

    transformed = vectorizer.transform([text])
    prediction = model.predict(transformed)

    print(f"\nPrediction: {prediction[0].upper()}")

# -----------------------------
# Add Data
# -----------------------------
def add_data():
    text = input("Enter news text: ")
    label = input("Enter label (real/fake): ").lower()

    if label not in ["real", "fake"]:
        print("Invalid label!")
        return

    data = load_data()
    new_row = pd.DataFrame([[text, label]], columns=["text", "label"])
    data = pd.concat([data, new_row], ignore_index=True)

    data.to_csv("news.csv", index=False)
    print("Data added successfully!")

# -----------------------------
# View Data
# -----------------------------
def view_data():
    data = load_data()
    print("\nDataset:")
    print(data)

# -----------------------------
# Menu
# -----------------------------
def menu():
    data = load_data()

    if len(data) < 2:
        print("Not enough data! Add more news first.")

    model, vectorizer = train_model(data)

    while True:
        print("\n--- Fake News Detection ---")
        print("1. Predict News")
        print("2. Add Data")
        print("3. View Data")
        print("4. Exit")

        choice = input("Enter choice: ")

        if choice == "1":
            predict_news(model, vectorizer)
        elif choice == "2":
            add_data()
            data = load_data()
            model, vectorizer = train_model(data)  # retrain
        elif choice == "3":
            view_data()
        elif choice == "4":
            print("Exiting...")
            break
        else:
            print("Invalid choice!")

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    menu()