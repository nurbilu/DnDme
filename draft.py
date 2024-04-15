import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import csv

# Extended example data: Sentences from D&D sessions
def load_sentences(filename='Extended_Sentences.csv'):
    sentences = []
    with open(filename, mode='r', newline='') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header row
        for row in csv_reader:
            sentences.append(row[1])  # Assuming the sentences are stored in the second column
    return sentences

# Load sentences from the file
sentences = load_sentences()

# Tokenize sentences
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
padded_sequences = pad_sequences(sequences, padding='post', maxlen=max(len(seq) for seq in sequences))

# Create a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(1000, 16, input_length=padded_sequences.shape[1]),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

# Explicitly building the model by specifying the input shape
model.build((None, padded_sequences.shape[1]))  # None allows the model to accept any batch size

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Show the model summary
model.summary()

