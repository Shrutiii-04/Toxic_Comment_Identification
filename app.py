from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import pickle
import os
import pandas as pd
from collections import Counter

app = Flask(__name__)

# ------------------- Load the trained model -------------------
model = tf.keras.models.load_model("model/toxicity.h5")

# ------------------- Load TextVectorization -------------------
vectorizer_path = "model/vectorizer.pkl"
if os.path.exists(vectorizer_path):
    with open(vectorizer_path, "rb") as f:
        vocab = pickle.load(f)
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=200000,
        output_sequence_length=1800,
        output_mode='int'
    )
    vectorizer.set_vocabulary(vocab)
else:
    print("Warning: vectorizer.pkl not found. Using temporary vectorizer (predictions may be inaccurate).")
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=200000,
        output_sequence_length=1800,
        output_mode='int'
    )
    vectorizer.adapt(["sample text"])

# ------------------- Toxic comment categories -------------------
labels = ["Toxic", "Severe Toxic", "Obscene", "Threat", "Insult", "Identity Hate"]

# ------------------- Home Route -------------------
@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    overall = None
    color = None
    comment = None
    predictions = None
    categories = None
    csv_results = False
    chart_labels = None
    chart_values = None

    if request.method == "POST":
        action = request.form.get("action")

        # ------------------- Single Comment Analysis -------------------
        if action == "analyze":
            comment = request.form.get("comment")
            if comment:
                vectorized = vectorizer([comment])
                preds = model.predict(vectorized)[0]
                result = dict(zip(labels, preds))
                overall_score = max(preds)
                overall = "Toxic" if overall_score > 0.7 else "Borderline" if overall_score > 0.3 else "Clean"
                color = "red" if overall_score > 0.7 else "orange" if overall_score > 0.3 else "green"
                predictions = [round(float(p), 4) for p in preds]
                categories = labels

        # ------------------- CSV Batch Analysis -------------------
        elif action == "upload_csv":
            file = request.files['csv_file']
            if file and file.filename.endswith('.csv'):
                df = pd.read_csv(file)
                possible_cols = ['comment', 'comment_text', 'text']
                text_col = None
                for col in possible_cols:
                    if col in df.columns:
                        text_col = col
                        break

                if text_col is None:
                    overall = "CSV must have a comment column like 'comment' or 'comment_text'"
                else:
                    # Create lists to store probabilities
                    all_probs = {label: [] for label in labels}
                    predicted_labels = []

                    for comment_text in df[text_col].fillna(""):
                        vectorized = vectorizer([comment_text])
                        preds = model.predict(vectorized)[0]
                        preds_rounded = [round(float(p), 4) for p in preds]

                        # Save probabilities
                        for i, label in enumerate(labels):
                            all_probs[label].append(preds_rounded[i])

                        # Determine overall label
                        max_pred = max(preds_rounded)
                        if max_pred < 0.3:
                            predicted_label = "Clean"
                        elif max_pred < 0.7:
                            predicted_label = "Borderline"
                        else:
                            predicted_label = "Toxic"
                        predicted_labels.append(predicted_label)

                    # Add columns to DataFrame
                    for label in labels:
                        df[label + "_probability"] = all_probs[label]
                    df['Predicted_Label'] = predicted_labels

                    # Save CSV
                    output_path = os.path.join("static", "analysis_results.csv")
                    os.makedirs("static", exist_ok=True)
                    df.to_csv(output_path, index=False)
                    csv_results = True
                    overall = f"Processed {len(df)} comments."

                    # ✅ FIXED Chart Data Section
                    label_counts = Counter(predicted_labels)
                    chart_labels = list(label_counts.keys())
                    chart_values = list(label_counts.values())

    return render_template(
        "index4.html",
        result=result,
        overall=overall,
        color=color,
        comment=comment,
        predictions=predictions,
        categories=categories,
        csv_results=csv_results,
        chart_labels=chart_labels,
        chart_values=chart_values
    )

# ------------------- Run App -------------------
if __name__ == "__main__":
    app.run(debug=True)
