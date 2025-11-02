import tensorflow as tf
import pandas as pd
import pickle
import os

# Load your Jigsaw dataset (adjust path if needed)
df = pd.read_csv("train.csv")  # the file you used for training
texts = df["comment_text"].astype(str).tolist()

# Create and adapt the vectorizer
vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=200000,
    output_sequence_length=1800
)
vectorizer.adapt(texts)

# Save vocabulary only (not the layer)
vocab = vectorizer.get_vocabulary()
os.makedirs("model", exist_ok=True)
with open("model/vectorizer.pkl", "wb") as f:
    pickle.dump(vocab, f)

print("✅ Vectorizer vocabulary saved to model/vectorizer.pkl")
