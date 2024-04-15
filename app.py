import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import csv
import random

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
    tf.keras.layers.Dense(len(sentences), activation='softmax')  # Output size is now the number of sentences
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Function to select a sentence based on the previous user input
def select_sentence(user_input):
    sequence = tokenizer.texts_to_sequences([user_input])
    padded = pad_sequences(sequence, maxlen=padded_sequences.shape[1], padding='post')
    prediction = model.predict(padded)
    # Return a random sentence as a response
    return sentences[random.randint(0, len(sentences) - 1)]

# Chat function to interact with the user
def chat():
    print("Welcome to the D&D sentence reaction AI!")
    ai_sentence = random.choice(sentences)
    print("AI starts with:", ai_sentence)
    while True:
        user_input = input("Your response (type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        ai_sentence = select_sentence(user_input)
        print("AI reacts with:", ai_sentence)

if __name__ == '__main__':
    chat()

