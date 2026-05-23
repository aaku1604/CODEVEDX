import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# -----------------------------
# Safe Input
# -----------------------------
def get_float_input(msg):
    while True:
        try:
            return float(input(msg))
        except ValueError:
            print("Invalid input! Enter a number.")

# -----------------------------
# Load Data
# -----------------------------
def load_data():
    try:
        return pd.read_csv("student_data.csv")
    except:
        return pd.DataFrame(columns=["attendance", "marks", "study_hours"])

# -----------------------------
# Save Data
# -----------------------------
def save_data(data):
    data.to_csv("student_data.csv", index=False)

# -----------------------------
# Add Data
# -----------------------------
def add_data():
    attendance = get_float_input("Enter attendance (%): ")
    marks = get_float_input("Enter marks: ")
    study_hours = get_float_input("Enter study hours: ")

    data = load_data()
    new_row = pd.DataFrame([[attendance, marks, study_hours]],
                           columns=["attendance", "marks", "study_hours"])
    
    data = pd.concat([data, new_row], ignore_index=True)
    save_data(data)

    print("Data added successfully!")

# -----------------------------
# View Data
# -----------------------------
def view_data():
    data = load_data()
    print("\nStudent Data:")
    print(data)

# -----------------------------
# Predict Performance
# -----------------------------
def predict_performance():
    data = load_data()

    if len(data) < 2:
        print("Not enough data!")
        return

    X = data[["attendance", "study_hours"]]
    y = data["marks"]

    model = LinearRegression()
    model.fit(X, y)

    attendance = get_float_input("Enter attendance: ")
    study_hours = get_float_input("Enter study hours: ")

    prediction = model.predict([[attendance, study_hours]])

    print(f"Predicted Marks: {prediction[0]:.2f}")

# -----------------------------
# Show Graph
# -----------------------------
def show_graph():
    data = load_data()

    if len(data) == 0:
        print("No data available!")
        return

    plt.scatter(data["study_hours"], data["marks"], color="blue")
    plt.xlabel("Study Hours")
    plt.ylabel("Marks")
    plt.title("Study Hours vs Marks")
    plt.show()

# -----------------------------
# Menu
# -----------------------------
def menu():
    while True:
        print("\n--- Student Performance Prediction ---")
        print("1. Add Data")
        print("2. View Data")
        print("3. Predict Marks")
        print("4. Show Graph")
        print("5. Exit")

        choice = input("Enter choice: ")

        if choice == "1":
            add_data()
        elif choice == "2":
            view_data()
        elif choice == "3":
            predict_performance()
        elif choice == "4":
            show_graph()
        elif choice == "5":
            print("Exiting...")
            break
        else:
            print("Invalid choice!")

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    menu()