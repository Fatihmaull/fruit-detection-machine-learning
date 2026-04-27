import kagglehub
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    # 1. Download & Load Data
    path = kagglehub.dataset_download("joshmcadams/oranges-vs-grapefruit")
    file_path = os.path.join(path, "citrus.csv")
    df = pd.read_csv(file_path)
    
    # Buat folder assets jika belum ada untuk menyimpan gambar
    if not os.path.exists('assets'):
        os.makedirs('assets')

    # 2. Visualisasi (Heatmap)
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.savefig('assets/heatmap.png')
    plt.close()

    # 3. Preprocessing
    le = LabelEncoder()
    df['name'] = le.fit_transform(df['name'])
    X = df.drop(columns=['name'])
    y = df['name']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 4. Modeling
    models = {
        "Decision Tree": DecisionTreeClassifier(),
        "Naive Bayes": GaussianNB(),
        "SVM": SVC()
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = accuracy_score(y_test, y_pred)
        print(f"Model {name} Accuracy: {results[name]:.4f}")

    # 5. Save Accuracy Graph
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(results.keys()), y=list(results.values()), palette='viridis')
    plt.title('Accuracy Comparison')
    plt.savefig('assets/graph.png')
    plt.close()

    # 6. Save Confusion Matrix (Best Model)
    best_model = models["Decision Tree"] # Berdasarkan riset sebelumnya
    y_pred_best = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Confusion Matrix - Decision Tree')
    plt.savefig('assets/conf.png')
    plt.close()

    print("\nProses Selesai. Semua visualisasi tersimpan di folder 'assets/'")

if __name__ == "__main__":
    main()
