"""
A script that generates chord progressions using an LSTM model trained on the Rock Corpus
dataset, developed in the context of the Valerio Velardo's "Computational Music Creativity"
course from the Master in Sound and Music Computing at Universitat Pompeu Fabra.

authors: Sergio Cárdenas Gracia & Siddharth Saxena
"""

import json
import numpy as np
import tensorflow.keras as keras
from preprocess import SEQUENCE_LENGTH, SONGS_MAPPING_PATH, CHORDS_MAPPING_PATH


SONGS_MODEL_PATH = "../models/LSTM-songs-50ep.h5"
CHORDS_MODEL_PATH = "../models/LSTM-chords-50ep.h5"


class ChordGenerator:
    """
    A class that wraps the LSTM model and offers utilities to generate chords.
    """

    def __init__(self, model_path, mapping_path):
        """
        Initializes the ChordGenerator with a trained LSTM model and mappings.

        Parameters:
            model_path (str): Path to the trained model file.

        Returns:
            None
        """
        self.model_path = model_path
        self.model = keras.models.load_model(model_path)

        with open(mapping_path, "r") as fp:
            self._mappings = json.load(fp)

        self._start_symbols = ["/"] * SEQUENCE_LENGTH


    def generate_chords(self, seed, num_steps, max_sequence_length, temperature):
        """
        Generates a chord sequence using the trained LSTM model.

        Parameters:
            seed (str): Chord seed with the notation used to encode the dataset.
            num_steps (int): Number of steps to generate.
            max_sequence_length (int): Maximum number of steps in the seed to consider for generation.
            temperature (float): A value in the interval [0, 1]. Values closer to 0 make the model
                more deterministic, while values closer to 1 make the generation more unpredictable.

        Returns:
            list of str: List of symbols representing the generated chord sequence.
        """
        # create seed with start symbols
        seed = seed.split()
        chord_sequence = seed
        seed = self._start_symbols + seed

        # Map seed to integer values
        seed = [self._mappings[symbol] for symbol in seed]

        for _ in range(num_steps):
            # Limit the seed to max_sequence_length
            seed = seed[-max_sequence_length:]

            # One-hot encode the seed
            onehot_seed = keras.utils.to_categorical(seed, num_classes=len(self._mappings))
            onehot_seed = onehot_seed[np.newaxis, ...] 

            # Make a prediction
            probabilities = self.model.predict(onehot_seed)[0]
            output_int = self._sample_with_temperature(probabilities, temperature)

            # Update seed
            seed.append(output_int)

            # Map integer back to symbol
            output_symbol = [k for k, v in self._mappings.items() if v == output_int][0]

            # Check if the chord sequence ends
            if output_symbol == "/":
                break

            # Append the generated symbol to the chord sequence
            chord_sequence.append(output_symbol)

        return chord_sequence


    def _sample_with_temperature(self, probabilites, temperature):
        """
        Samples an index from a probability array, reapplying softmax using temperature.

        Parameters:
            probabilites (nd.array): Array containing probabilities for each of the possible outputs.
            temperature (float): A value in the interval [0, 1]. Values closer to 0 make the model
                more deterministic, while values closer to 1 make the generation more unpredictable.

        Returns:
            int: Selected output index based on the temperature-scaled probabilities.
        """
        predictions = np.log(probabilites) / temperature
        probabilites = np.exp(predictions) / np.sum(np.exp(predictions))

        choices = range(len(probabilites))
        index = np.random.choice(choices, p=probabilites)

        return index


def generate_chords(model, mapping_path, seed, num_steps, max_sequence_length, temperature):

    chord_song_generator = ChordGenerator(model, mapping_path)

    generated_chords = chord_song_generator.generate_chords(
        seed=seed,
        num_steps=num_steps,
        max_sequence_length=max_sequence_length,
        temperature=temperature
    )

    return generated_chords


if __name__ == "__main__":
    # Generate chord sequence for a rock song
    song_seed = ""
    bars_to_generate = 50
    temperature = 0.8
    generated_song = generate_chords(SONGS_MODEL_PATH, SONGS_MAPPING_PATH, song_seed, 5*bars_to_generate, SEQUENCE_LENGTH, temperature)

    # Generate rock chord progression
    chords_seed = ""
    chords_to_generate = 50
    temperature = 0.8
    generated_chords = generate_chords(CHORDS_MODEL_PATH, CHORDS_MAPPING_PATH, chords_seed, chords_to_generate, SEQUENCE_LENGTH, temperature)
    
    # Show results
    print(f"Generated Song:\n{' '.join(generated_song)}")
    print(f"Generated Chord Progression:\n{' | '.join(generated_chords)}")
