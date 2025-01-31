"""
An LSTM model training script for the Rock Corpus dataset, developed in the
context of the Valerio Velardo's "Computational Music Creativity" course
from the Master in Sound and Music Computing at Universitat Pompeu Fabra.

authors: Sergio Cárdenas Gracia & Siddharth Saxena
"""

import tensorflow.keras as keras
from preprocess import generate_training_sequences, SEQUENCE_LENGTH, SONGS_DATASET_FILE, CHORDS_DATASET_FILE, SONGS_MAPPING_PATH, CHORDS_MAPPING_PATH


SONGS_OUTPUT_UNITS = 136
CHORDS_OUTPUT_UNITS = 133
NUM_UNITS = [256]
LOSS = "sparse_categorical_crossentropy"
LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 64
SONGS_MODEL_PATH = "../models/LSTM-songs-50ep.h5"
CHORDS_MODEL_PATH = "../models/LSTM-chords-50ep.h5"


def build_model(output_units, num_units, loss, learning_rate):
    """
    Builds, compiles, and returns an LSTM-based Keras model.

    Parameters:
        output_units (int): The number of units in the output layer.
        num_units (list of int): A list specifying the number of units in the LSTM hidden layers.
        loss (str): The loss function to use.
        learning_rate (float): The learning rate for the optimizer.

    Returns:
        keras.Model: The compiled Keras model.
    """
    # Create the model architecture
    input = keras.layers.Input(shape=(None, output_units))
    x = keras.layers.LSTM(num_units[0])(input)
    x = keras.layers.Dropout(0.2)(x)

    output = keras.layers.Dense(output_units, activation="softmax")(x)

    model = keras.Model(input, output)

    # Compile the model
    model.compile(loss=loss,
                  optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics=["accuracy"])

    model.summary()

    return model


def train(output_units, dataset_file, mapping_path, model_path, num_units=NUM_UNITS, loss=LOSS, learning_rate=LEARNING_RATE):
    """
    Trains the model on the generated training sequences and saves it to the specified path.

    Parameters:
        output_units (int): The number of units in the output layer.
        num_units (list of int): A list specifying the number of units in the LSTM hidden layers.
        loss (str): The loss function to use.
        learning_rate (float): The learning rate for the optimizer.

    Returns:
        None
    """

    # Generate the training sequences
    inputs, targets = generate_training_sequences(SEQUENCE_LENGTH, dataset_file, mapping_path)

    # Build the network
    model = build_model(output_units, num_units, loss, learning_rate)

    # Train the model
    model.fit(inputs, targets, epochs=EPOCHS, batch_size=BATCH_SIZE)

    # Save the model
    model.save(model_path)


if __name__ == "__main__":
    train(SONGS_OUTPUT_UNITS, SONGS_DATASET_FILE, SONGS_MAPPING_PATH, SONGS_MODEL_PATH)
    train(CHORDS_OUTPUT_UNITS, CHORDS_DATASET_FILE, CHORDS_MAPPING_PATH, CHORDS_MODEL_PATH)
