[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/MMXgaRe2)
# Assignment 3: Rock your LSTM

Hey Team,

First off, fantastic job on the LSTM for melody generation\! I hope you learned a lot from that project. As you know, we’re pivoting to focus on chord generation instead of melody generation. I know this might feel a bit frustrating, but welcome to startup life—it’s all about adapting and seizing new opportunities. And hey, you’re going to learn a ton from this shift\!

The good news is that all the hard work you did on the LSTM for melody generation won’t go to waste. We can re-adapt it for our new mission: **coding an LSTM to automatically generate sequences of chords for rock music.**

## The Plan

Luckily, we’ve got an amazing dataset to work with: [Rock Corpus Version 1.1](https://rockcorpus.midside.com/). This dataset includes **200 Billboard songs annotated by two experienced music theorists**. You can learn more about the dataset and the [harmonic analysis here](https://rockcorpus.midside.com/harmonic_analyses.html).

* Chords are annotated using **Roman numeral analysis** (learn more [here](https://en.wikipedia.org/wiki/Roman_numeral_analysis)).  
* Chord sequences are organized by **chorus, verse, and bridge**, giving us structure to work with.

Here’s what I suggest for the next steps:

1. **Re-adapt the LSTM Melody Generator:**  
   * Modify the model so it generates **chord sequences** instead of melodies.  
   * Train the LSTM on the **sequence of Roman numerals**, generating one Roman numeral at a time.  
2. **Redesign the Architecture (Optional):**  
   * If you think changing the LSTM architecture will improve performance (e.g., changing the number of layers / neurons), go for it\! Experimentation is encouraged.  
3. **Update the Data Preprocessors:**  
   * Adapt the data preprocessing pipeline to prepare the chord sequences from the Rock Corpus dataset in a format the LSTM can ingest.

## Deliverables

Here’s what I expect by the end of this sprint:

1. **Training Module:**  
   * A module I can run to train the LSTM on the Rock Corpus dataset.  
   * Make sure the dataset preprocessing code is included, and document how to use it.  
2. **Generation Module:**  
   * A model I can run to generate a chord sequence, where I can specify the **number of chords** in the sequence.  
   * Output should be a simple string of Roman numerals, like this:  
     `I | ii7 | V | I`

## Optional Stretch Goal

If you’re feeling ambitious, consider this enhancement:

* Train separate models for **verse-only** and **chorus-only** chord sequences.  
  * Verses and choruses likely have distinct chord patterns, so this could lead to more stylistically accurate results.  
  * If this feels like too much for now, it’s perfectly fine to train on all sequences at once and ignore the structural distinctions.

I’m really excited to see where this pivot takes us. The work you’re doing is laying the foundation for something truly impactful in the music tech space. Have fun with this challenge, and remember—rock on\! 🎸🎵

Valerio, your CTO
