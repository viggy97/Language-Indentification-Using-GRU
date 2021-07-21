# Language-Indentification-Using-GRU

This project classifies whether the language spoken in a given audio file is English, Hindi or Mandarin.

# GRU Model

model.py defined a GRU model for the classification problems along with the audio data preprocessing actions.

# Data:

    We used recorded 10-15min audio files (.mp3) in the above 3 languages and place them in sperate folders within a single folder. (eg: ../train/train_english for english audio files and similarly for the other languages)
    
# Data format requirements
    These audio files must be in .wav format with 16 kHz sampling rate and mono format (single channel). This can be done using open source software like Sox. Steps  
    
        (for ubuntu):

        install sox by: " sudo apt-get install sox"

        type the following command on terminal for converting a given audio.mp3 file into mono.wav file with above specs:

        "sox audio.mp3 -c 1 -r 16k -t wav mono.wav"
        
# Preprocessing

We use librosa python package to extract the MFCC features from the Audio signals.

