from sklearn.svm import SVC
import joblib
import numpy as np

embeddings, labels = joblib.load("embeddings.pkl")

print("Training SVM...")

model = SVC(kernel='linear', probability=True)
model.fit(embeddings, labels)

joblib.dump(model, "face_model.pkl")
print("Model saved as face_model.pkl")
