import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import os
import pickle

# Load CSV with error handling
df = pd.read_csv(
    "data/Text_Emotion.csv",
    quoting=1,
    escapechar='\\',
    encoding='utf-8',
    on_bad_lines='skip'
)

# Preview first few rows
print("Loaded Data Sample:\n", df.head())

# Use actual column names: text and label
df.columns = df.columns.str.strip()  # Remove whitespace if any
if 'text' in df.columns and 'label' in df.columns:
    df['content'] = df['text']
    df['sentiment'] = df['label']
else:
    raise ValueError("Expected columns 'text' and 'label' not found in CSV.")

# Encode sentiment labels
le = LabelEncoder()
df['encoded_label'] = le.fit_transform(df['sentiment'])

# Tokenize text
tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
tokenizer.fit_on_texts(df['content'])

X = tokenizer.texts_to_sequences(df['content'])
X = pad_sequences(X, maxlen=100)

y = to_categorical(df['encoded_label'])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model
model = Sequential([
    Embedding(5000, 128, input_length=100),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dropout(0.2),
    Dense(y.shape[1], activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))

# Save model and encoder
os.makedirs("models", exist_ok=True)
model.save("models/emotion_model.h5")

with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("✅ Model training complete. Model and label encoder saved.")
