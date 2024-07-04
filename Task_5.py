import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

# Load dataset
# Assuming 'data.txt' contains the preprocessed handwritten text data
with open('data.txt', 'r') as file:
    text = file.read()

# Character encoding
chars = sorted(list(set(text)))
char_to_int = {c: i for i, c in enumerate(chars)}
int_to_char = {i: c for i, c in enumerate(chars)}

# Prepare input-output pairs
seq_length = 100
X = []
y = []
for i in range(0, len(text) - seq_length):
    seq_in = text[i:i + seq_length]
    seq_out = text[i + seq_length]
    X.append([char_to_int[char] for char in seq_in])
    y.append(char_to_int[seq_out])

# Reshape and normalize the data
X = np.reshape(X, (len(X), seq_length, 1))
X = X / float(len(chars))
y = to_categorical(y)

# Build the RNN model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(LSTM(256))
model.add(Dense(y.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

# Define checkpoint callback
filepath = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# Train the model
model.fit(X, y, epochs=20, batch_size=128, callbacks=callbacks_list)

# Load the network weights
filename = "weights-improvement-20-1.2345.hdf5"  # Replace with the best checkpoint filename
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Function to generate text
def generate_text(model, length, seed_text, int_to_char, char_to_int):
    result = []
    input_text = seed_text

    for _ in range(length):
        input_seq = [char_to_int[char] for char in input_text]
        input_seq = np.reshape(input_seq, (1, len(input_seq), 1))
        input_seq = input_seq / float(len(chars))

        prediction = model.predict(input_seq, verbose=0)
        index = np.argmax(prediction)
        result.append(int_to_char[index])

        input_text = input_text[1:] + int_to_char[index]

    return ''.join(result)

# Generate new text
seed_text = "the quick brown fox jumps over the lazy dog"
generated_text = generate_text(model, 500, seed_text, int_to_char, char_to_int)
print(generated_text)
