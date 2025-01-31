"""
A preprocessing script for the Rock Corpus dataset to prepare data for training an LSTM
model, developed in the context of the Valerio Velardo's "Computational Music Creativity"
course from the Master in Sound and Music Computing at Universitat Pompeu Fabra.

authors: Sergio Cárdenas Gracia & Siddharth Saxena
"""

import os
import re
import json
import subprocess
import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt


ROCK_CORPUS_PATH = "../data/raw/rock_corpus_v1-1"
SONGS_SAVE_DIR = "../data/processed/rock_songs"
CHORDS_SAVE_DIR = "../data/processed/rock_chords"
SONGS_DISTRIBUTION_PLOT_PATH = "../data/processed/songs_distribution.png"
CHORDS_DISTRIBUTION_PLOT_PATH = "../data/processed/chords_distribution.png"
SONGS_DATASET_FILE = "../data/training/rock_songs_dataset"
CHORDS_DATASET_FILE = "../data/training/rock_chords_dataset"
SONGS_MAPPING_PATH = "../data/training/rock_songs_mapping.json"
CHORDS_MAPPING_PATH = "../data/training/rock_chords_mapping.json"
SEQUENCE_LENGTH = 32


def correct_measure(song):
    """
    Checks if a song has all measures in 4/4 time signature using the expand6 script.

    Parameters:
        song (str): Path to the song file.

    Returns:
        bool: True if all measures are in 4/4, False otherwise.
    """
    measures_analysis = subprocess.run(['./expand6', '-v', str(-1), song], 
                                        capture_output=True, text=True)
    measures = measures_analysis.stdout
    measures = [measure for measure in measures.split() if measures.strip()][1::2]
    if all(element == '404' for element in measures):
        return True
    return False


def has_key_changes(song):
    """
    Checks if a song contains any key changes using the expand6 script.

    Parameters:
        song (str): Path to the song file.

    Returns:
        bool: True if there are key changes, False otherwise.
    """
    keys_analysis = subprocess.run(['./expand6', '-v', str(-2), song], 
                                    capture_output=True, text=True)
    keys = keys_analysis.stdout
    keys = [key for key in keys.split() if keys.strip()]
    if len(keys) != 2:
        return True
    return False
    

def extract_chords(song):
    """
    Extracts the chord sequence from a song by removing metadata.

    Parameters:
        song (str): Song string with metadata.

    Returns:
        str: Chord sequence without metadata.
    """
    bracket_index = song.find("] ")
    if bracket_index != -1:
        return song[bracket_index + 1:].strip()
    return song


def expand_song(song):
    """
    Expands a song into its chord sequence using the expand6 script.

    Parameters:
        song (str): Path to the song file.

    Returns:
        str: Extracted chord sequence of the song.
    """
    one_line_analysis = subprocess.run(['./expand6', '-v', str(1), song], 
                                        capture_output=True, text=True)
    song = one_line_analysis.stdout
    return extract_chords(song)


def load_songs_in_rock_corpus(dataset_path):
    """
    Loads all valid songs in the rock corpus dataset, filtering out songs 
    with incorrect measures or key changes.

    Parameters:
        dataset_path (str): Path to the dataset.

    Returns:
        list: List of chord sequences for all valid songs.
    """
    expanded_songs = []

    # Iterate through all files in the dataset
    for path, _, files in os.walk(dataset_path):
        for file in files:
            # Only process text files
            if file[-3:] == "txt":
                file_path = os.path.join(path, file)

                # Filter for songs with valid measures and no key changes
                if correct_measure(file_path) and not has_key_changes(file_path):
                    expanded_song = expand_song(file_path)
                    expanded_songs.append(expanded_song)

    return expanded_songs


def correct_number_of_chords_per_bar(song):
    """
    Verifies that each bar in the song contains an acceptable number of chords.

    Parameters:
        song (str): A string representation of the song with chords and bars.

    Returns:
        bool: True if all bars have 0, 1, 2, or 4 chords, False otherwise.
    """
    bars = song.split("|")
    for bar in bars:
        elements = [elem for elem in bar.split() if elem.strip()]
        if len(elements) not in [0, 1, 2, 4]:
            return False
    return True


def encode_song(song):
    """
    Converts the string of chords in a song into a time series. Each item in the
    encoded list represents a quarter length. The symbols used are:
        - Roman numerals for chords
        - 'R' for rests
        - '.' for sustained notes or rests
        - '|' for bar splits

    Parameters:
        song (str): A string representation of the song with chords and bars.

    Returns:
        str: Encoded string representation of the song.
    """
    encoded_song = []
    bars = song.split("|")

    for bar in bars:
        bar = bar.strip()  # Remove leading/trailing spaces
        elements = bar.split()

        # Encode chords/rests based on the number of elements in the bar
        if len(elements) == 2:
            encoded_song.extend([elements[0], ".", elements[1], "."])
        else:
            encoded_song.extend(elements + ["."] * (4 - len(elements)))

        encoded_song.append("|")  # Append bar split symbol

    # Remove the trailing bar split if it exists
    if encoded_song and encoded_song[-1] == "|":
        encoded_song.pop()

    # Convert encoded song list to a single string
    encoded_song = " ".join(map(str, encoded_song))

    return encoded_song


def encode_chords(song):
    """
    Converts the string of chords in a song into a chord progression represented
    with roman numerals. Chord durations are not taken into account.

    Parameters:
        song (str): A string representation of the song with chords and bars.

    Returns:
        str: Encoded string representation of chord sequence.
    """    
    elements = song.split()

    # Isolate roman numerals
    characters_to_remove = {"R", ".", "|"}
    encoded_chord_seq = [item for item in elements if item not in characters_to_remove]

    # Convert encoded chord sequence to a single string
    encoded_chord_seq = " ".join(map(str, encoded_chord_seq))

    return encoded_chord_seq


def preprocess(dataset_path):
    """
    Preprocesses the dataset by encoding valid songs and chord progressions and 
    saving them to SONGS_SAVE_DIR and CHORDS_SAVE_DIR, respectively.

    Parameters:
        dataset_path (str): Path to the rock corpus dataset.
    """
    print("Loading songs...")
    songs = load_songs_in_rock_corpus(dataset_path)
    print(f"Loaded {len(songs)} songs.")

    encoded_songs = []
    encoded_chords = []

    for i, song in enumerate(songs):
        # Skip songs with invalid chord structures
        if not correct_number_of_chords_per_bar(song):
            continue

        encoded_song = encode_song(song)
        encoded_songs.append(encoded_song)

        # Save encoded song to a file
        save_path = os.path.join(SONGS_SAVE_DIR, str(i))
        with open(save_path, "w") as fp:
            fp.write(encoded_song)

        encoded_chord_seq = encode_chords(song)
        encoded_chords.append(encoded_chord_seq)

        # Save encoded chord sequence to a file
        save_path = os.path.join(CHORDS_SAVE_DIR, str(i))
        with open(save_path, "w") as fp:
            fp.write(encoded_chord_seq)

    print(f"Encoded {len(encoded_songs)} songs.")


def load(file_path):
    """
    Reads and returns the contents of a file.

    Parameters:
        file_path (str): Path to the file to be read.

    Returns:
        str: The content of the file as a string.
    """
    with open(file_path, "r") as fp:
        song = fp.read()
    return song


def create_single_file_dataset(dataset_path, file_dataset_path, sequence_length):
    """
    Generates a single file containing all encoded elements with delimiters separating them.

    Parameters:
        dataset_path (str): Path to the folder containing the encoded elements.
        file_dataset_path (str): Path where the combined dataset file will be saved.
        sequence_length (int): Number of time steps to be considered for training, 
        determines delimiter length.

    Returns:
        str: A single string containing all elements from the dataset with delimiters.
    """
    new_element_delimiter = "/ " * sequence_length
    elements = ""

    # Load encoded elements and add delimiters
    for path, _, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(path, file)
            element = load(file_path)
            elements = elements + element + " " + new_element_delimiter

    # Remove the trailing space from the last delimiter
    elements = elements[:-1]

    # Save the combined dataset to the specified file
    with open(file_dataset_path, "w") as fp:
        fp.write(elements)

    return elements


def create_mapping(elements, mapping_path):
    """
    Creates a JSON file mapping unique symbols in the given dataset to integers.

    Parameters:
        elements (str): A single string containing all elements in the dataset.
        mapping_path (str): Path where the mapping JSON file will be saved.

    Returns:
        None
    """
    mappings = {}

    # Identify the vocabulary (unique symbols in the dataset)
    elements = elements.split()
    vocabulary = list(set(elements))

    # Create mappings from symbols to integers
    for i, symbol in enumerate(vocabulary):
        mappings[symbol] = i

    # Save the vocabulary mappings to a JSON file
    with open(mapping_path, "w") as fp:
        json.dump(mappings, fp, indent=4)


def get_elements_distribution(elements):
    """
    Calculate the frequency distribution of elements in a dataset.
    
    Parameters:
        elements (str): Space-separated string of elements.
    
    Returns:
        dict: A dictionary where keys are elements and values are their counts.
    """
    distribution = {}
    for element in elements.split():
        distribution[element] = distribution.get(element, 0) + 1
    return distribution


def plot_distribution(distribution, plot_title, output_path, threshold=200, xlabel="Elements", ylabel="Counts"):
    """
    Plot a distribution of elements with aggregation for low-frequency items.
    
    Parameters:
        distribution (dict): The raw element frequency distribution.
        plot_title (str): Title of the plot.
        output_path (str): Path to save the resulting plot image.
        threshold (int): Minimum count for an element to appear separately (default: 200).
        xlabel (str): Label for the x-axis (default: "Elements").
        ylabel (str): Label for the y-axis (default: "Counts").
    """
    # Exclude "/" delimiter
    distribution = {k: v for k, v in distribution.items() if k not in "/"}

    # Aggregate low-frequency elements into "others"
    aggregated_distribution = {}
    others_count = 0
    for element, count in distribution.items():
        if count < threshold:
            others_count += count
        else:
            aggregated_distribution[element] = count
    if others_count > 0:
        aggregated_distribution["others"] = others_count

    # Sort distribution by frequency for better visualization
    aggregated_distribution = dict(sorted(aggregated_distribution.items(), key=lambda x: x[1], reverse=True))

    # Prepare data for plotting
    labels = list(aggregated_distribution.keys())
    values = list(aggregated_distribution.values())

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.bar(labels, values, color='skyblue', edgecolor='black')
    plt.title(plot_title, fontsize=16, fontweight="bold")
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.xticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to: {output_path}")


def convert_elements_to_int(elements, mapping_path):
    """
    Converts a string of elements into a list of integers based on a mapping.

    Parameters:
        elements (str): A single string containing all elements.

    Returns:
        list: A list of integers representing the elements.
    """
    int_elements = []

    # Load mappings from the mapping file
    with open(mapping_path, "r") as fp:
        mappings = json.load(fp)

    # Transform songs string into a list of symbols
    elements = elements.split()

    # Map each symbol to its corresponding integer
    for symbol in elements:
        int_elements.append(mappings[symbol])

    return int_elements


def generate_training_sequences(sequence_length, dataset_file, mapping_path):
    """
    Creates input and output data samples (sequences of integers) for training.

    Parameters:
        sequence_length (int): Length of each input sequence.

    Returns:
        tuple: A tuple containing:
            - inputs (ndarray): One-hot encoded training inputs of shape 
            (# of sequences, sequence length, vocabulary size).
            - targets (ndarray): Training targets as a NumPy array.
    """
    # Load elements and map them to integers
    elements = load(dataset_file)
    int_elements = convert_elements_to_int(elements, mapping_path)

    inputs = []
    targets = []

    # Generate training sequences
    num_sequences = len(int_elements) - sequence_length
    for i in range(num_sequences):
        inputs.append(int_elements[i:i + sequence_length])
        targets.append(int_elements[i + sequence_length])

    # One-hot encode the input sequences
    vocabulary_size = len(set(int_elements))
    inputs = keras.utils.to_categorical(inputs, num_classes=vocabulary_size)
    targets = np.array(targets)

    print(f"There are {len(inputs)} sequences.")

    return inputs, targets


def main():
    # Preprocess the dataset
    preprocess(ROCK_CORPUS_PATH)

    # Create a single-file dataset from the preprocessed data
    songs = create_single_file_dataset(SONGS_SAVE_DIR, SONGS_DATASET_FILE, SEQUENCE_LENGTH)
    chords = create_single_file_dataset(CHORDS_SAVE_DIR, CHORDS_DATASET_FILE, SEQUENCE_LENGTH)

    # Create and save mappings for training
    create_mapping(songs, SONGS_MAPPING_PATH)
    create_mapping(chords, CHORDS_MAPPING_PATH)

    # Analyze and visualize dataset distribution
    songs_distribution = get_elements_distribution(songs)
    chords_distribution = get_elements_distribution(chords)

    plot_distribution(
        songs_distribution,
        plot_title="Distribution of elements in songs", 
        output_path=SONGS_DISTRIBUTION_PLOT_PATH,
        threshold=200,
        xlabel="Elements",
        ylabel="Counts"
        )
    
    plot_distribution(
        chords_distribution,
        plot_title="Distribution of chords in dataset",
        output_path=CHORDS_DISTRIBUTION_PLOT_PATH,
        threshold=200, 
        xlabel="Chords", 
        ylabel="Counts"
        )


if __name__ == "__main__":
    main()
