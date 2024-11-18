import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import re

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Parameters
max_features = 20000  # Increased vocabulary size
maxlen = 250  # Increased sequence length
embedding_dims = 256  # Increased embedding dimensions
batch_size = 64

# Load the IMDB dataset
print("Loading IMDB dataset...")
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# Pad sequences to ensure uniform length
print("Preprocessing data...")
x_train = pad_sequences(x_train, maxlen=maxlen, padding='post', truncating='post')
x_test = pad_sequences(x_test, maxlen=maxlen, padding='post', truncating='post')

# Calculate class weights to handle imbalance
total_samples = len(y_train)
n_positive = sum(y_train)
n_negative = total_samples - n_positive
class_weight = {
    0: total_samples / (2 * n_negative),
    1: total_samples / (2 * n_positive)
}

# Build an improved model
print("Building model...")
model = Sequential([
    Embedding(max_features, embedding_dims, input_length=maxlen),
    SpatialDropout1D(0.2),
    Bidirectional(LSTM(128, return_sequences=True)),
    Bidirectional(LSTM(64, return_sequences=True)),
    Bidirectional(LSTM(32)),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

# Compile with a lower learning rate
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])

print(model.summary())

# Define early stopping callback with more patience
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# Train the model with class weights and more epochs
print("\nTraining model...")
history = model.fit(x_train, y_train,
                   batch_size=batch_size,
                   epochs=15,
                   validation_split=0.2,
                   callbacks=[early_stopping],
                   class_weight=class_weight,
                   verbose=1)

# Evaluate the model
print("\nEvaluating model...")
loss, accuracy = model.evaluate(x_test, y_test)
print(f"\nTest accuracy: {accuracy:.4f}")
print(f"Test loss: {loss:.4f}")

# Save the model
print("\nSaving model...")
model.save('sentiment_analysis_model')

def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def predict_sentiment(text, word_index=imdb.get_word_index()):
    # Clean the input text
    text = clean_text(text)
    
    # Reverse word index to get words from indices
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    
    # Convert text to sequence
    words = text.split()
    sequence = []
    for word in words:
        if word in word_index and word_index[word] < max_features:
            sequence.append(word_index[word])
    
    # Pad sequence
    sequence = pad_sequences([sequence], maxlen=maxlen, padding='post', truncating='post')
    
    # Predict
    prediction = model.predict(sequence)[0][0]
    confidence = prediction if prediction > 0.5 else 1 - prediction
    return "Positive" if prediction > 0.5 else "Negative", confidence

# Test the model with sample positive and negative texts
positive_text = "This movie was really great! I enjoyed every moment of it."
negative_text = "This movie was terrible. Complete waste of time and money. The acting was horrible."

print("\nTesting with sample texts:")
sentiment, confidence = predict_sentiment(positive_text)
print(f"\nPositive sample: {positive_text}")
print(f"Predicted sentiment: {sentiment} (confidence: {confidence:.4f})")

sentiment, confidence = predict_sentiment(negative_text)
print(f"\nNegative sample: {negative_text}")
print(f"Predicted sentiment: {sentiment} (confidence: {confidence:.4f})")

# Additional test cases
test_cases = [
    "The worst movie I've ever seen!",
    "I absolutely hated everything about this.",
    "Don't waste your money on this garbage.",
    "Brilliant performance and amazing story!",
    "Such a disappointment, terrible plot."
]

print("\nAdditional test cases:")
for text in test_cases:
    sentiment, confidence = predict_sentiment(text)
    print(f"\nText: {text}")
    print(f"Predicted sentiment: {sentiment} (confidence: {confidence:.4f})")
