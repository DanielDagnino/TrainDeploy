import os
from path import Path
from pydub import AudioSegment


def convert_mp3_to_wav(mp3_path, wav_path):
    # Load the MP3 file using AudioSegment
    audio = AudioSegment.from_file(mp3_path, format='mp3')

    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(wav_path), exist_ok=True)

    # Construct the output WAV file path
    wav_path = os.path.splitext(wav_path)[0] + '.wav'

    # Export the audio as WAV
    audio.export(wav_path, format='wav')


def convert_directory(source_dir, target_dir):
    # Iterate over files and subdirectories in the source directory
    for root, _, files in os.walk(source_dir):
        for file in files:
            # Check if the file is an MP3
            if file.lower().endswith('.mp3'):
                # Get the relative path of the MP3 file
                relative_path = os.path.relpath(root, source_dir)
                mp3_path = os.path.join(root, file)

                # Construct the target WAV file path
                wav_path = os.path.join(target_dir, relative_path, file)

                # Convert the MP3 file to WAV and save in the target directory
                if not Path(wav_path).exists():
                    try:
                        convert_mp3_to_wav(mp3_path, wav_path)
                    except:
                        print(mp3_path)


# Usage example
source_directory = '/home/razor/MyData/FMA_Dataset_For_Music_Analysis/fma_medium'
target_directory = '/home/razor/MyData/FMA_Dataset_For_Music_Analysis/fma_medium_wav'

convert_directory(source_directory, target_directory)
