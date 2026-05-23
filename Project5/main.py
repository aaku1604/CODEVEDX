import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Load Data
# -----------------------------
def load_data():
    try:
        return pd.read_csv("data.csv")
    except:
        print("Error loading data!")
        return pd.DataFrame(columns=["user", "item", "rating"])

# -----------------------------
# Create User-Item Matrix
# -----------------------------
def create_matrix(data):
    return data.pivot_table(index="user", columns="item", values="rating").fillna(0)

# -----------------------------
# Recommend Items
# -----------------------------
def recommend_items():
    data = load_data()

    if len(data) == 0:
        print("No data available!")
        return

    matrix = create_matrix(data)

    similarity = cosine_similarity(matrix)
    similarity_df = pd.DataFrame(similarity, index=matrix.index, columns=matrix.index)

    user_id = int(input("Enter user ID: "))

    if user_id not in similarity_df.index:
        print("User not found!")
        return

    similar_users = similarity_df[user_id].sort_values(ascending=False)[1:]

    print("\nSimilar Users:")
    print(similar_users)

    top_user = similar_users.idxmax()

    recommended_items = matrix.loc[top_user]
    recommended_items = recommended_items[matrix.loc[user_id] == 0]

    print("\nRecommended Items:")
    print(recommended_items[recommended_items > 0])

# -----------------------------
# Add Data
# -----------------------------
def add_data():
    user = int(input("Enter user ID: "))
    item = input("Enter item name: ")
    rating = float(input("Enter rating (1-5): "))

    data = load_data()
    new_row = pd.DataFrame([[user, item, rating]],
                           columns=["user", "item", "rating"])
    
    data = pd.concat([data, new_row], ignore_index=True)
    data.to_csv("data.csv", index=False)

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
    while True:
        print("\n--- Recommendation System ---")
        print("1. Add Data")
        print("2. View Data")
        print("3. Get Recommendations")
        print("4. Exit")

        choice = input("Enter choice: ")

        if choice == "1":
            add_data()
        elif choice == "2":
            view_data()
        elif choice == "3":
            recommend_items()
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