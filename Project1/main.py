import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# -----------------------------
# Safe Input Function
# -----------------------------
def get_float_input(message):
    while True:
        try:
            return float(input(message))
        except ValueError:
            print("Invalid input! Please enter a number.")

# -----------------------------
# Load data
# -----------------------------
def load_data():
    try:
        data = pd.read_csv("data.csv")
    except:
        data = pd.DataFrame(columns=["hours", "usage"])
    return data

# -----------------------------
# Save data
# -----------------------------
def save_data(data):
    data.to_csv("data.csv", index=False)

# -----------------------------
# Add new data
# -----------------------------
def add_data():
    hours = get_float_input("Enter hours used: ")
    usage = get_float_input("Enter utility usage: ")
    
    data = load_data()
    new_row = pd.DataFrame([[hours, usage]], columns=["hours", "usage"])
    data = pd.concat([data, new_row], ignore_index=True)
    
    save_data(data)
    print("Data added successfully!")

# -----------------------------
# View data
# -----------------------------
def view_data():
    data = load_data()
    print("\nCurrent Data:")
    print(data)

# -----------------------------
# Predict usage
# -----------------------------
def predict_usage():
    data = load_data()
    
    if len(data) < 2:
        print("Not enough data to train model!")
        return
    
    X = data[["hours"]]
    y = data["usage"]
    
    model = LinearRegression()
    model.fit(X, y)
    
    hours = get_float_input("Enter hours to predict usage: ")
    prediction = model.predict([[hours]])
    
    print(f"Predicted usage: {prediction[0]:.2f}")

# -----------------------------
# Show Graph
# -----------------------------
def show_graph():
    data = load_data()
    
    if len(data) == 0:
        print("No data available!")
        return
    
    plt.scatter(data["hours"], data["usage"], color="blue", label="Data Points")
    
    if len(data) >= 2:
        model = LinearRegression()
        model.fit(data[["hours"]], data["usage"])
        predicted = model.predict(data[["hours"]])
        plt.plot(data["hours"], predicted, color="red", label="Regression Line")
    
    plt.xlabel("Hours")
    plt.ylabel("Usage")
    plt.title("Utility Usage Prediction")
    plt.legend()
    plt.show()

# -----------------------------
# Menu
# -----------------------------
def menu():
    while True:
        print("\n--- Utility Usage Prediction Tool ---")
        print("1. Add Data")
        print("2. View Data")
        print("3. Predict Usage")
        print("4. Show Graph")
        print("5. Exit")
        
        choice = input("Enter choice: ")
        
        if choice == "1":
            add_data()
        elif choice == "2":
            view_data()
        elif choice == "3":
            predict_usage()
        elif choice == "4":
            show_graph()
        elif choice == "5":
            print("Exiting...")
            break
        else:
            print("Invalid choice!")

# -----------------------------
# Run program
# -----------------------------
if __name__ == "__main__":
    menu()