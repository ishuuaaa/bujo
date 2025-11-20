
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

training_data = {
    "dbms": [
        "normalization keys relational model acid properties transaction",
        "schemas tables attributes er diagram database"
    ],
    "os": [
        "process scheduling deadlock threads memory paging segmentation",
        "cpu scheduling algorithms semaphore"
    ],
    "coa": [
        "pipelines instruction cycle cache mapping alu control unit",
        "number system microarchitecture"
    ],
    "dsa": [
        "stack queue linked list trees graphs sorting searching",
        "time complexity algorithms"
    ],
    "ai": [
        "search algorithms machine learning neural network agent",
        "alpha beta pruning a star algorithm"
    ]
}

texts, labels = [], []

for label, samples in training_data.items():
    for sample in samples:
        texts.append(sample)
        labels.append(label)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('model', MultinomialNB())
])

pipeline.fit(texts, labels)

with open("model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("Model trained and saved!")
