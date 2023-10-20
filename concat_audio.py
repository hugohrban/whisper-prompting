import wave
from pydub import AudioSegment
import numpy as np

lang = "en"
dir = f"{lang}_fleurs"
seed = 69
np.random.seed(seed)
indices = np.arange(100)
np.random.shuffle(indices)
infiles = [f"{dir}/audio_{i}.wav" for i in indices]
outfile = f"{dir}/all_concatenated.wav"

targets = []
with open(f"{dir}/all_transcripts_golden_{lang}.txt", "r") as f:
    for line in f:
        targets.append(line.strip())

with open(f"{dir}/all_transcripts_golden_{lang}_shuffled.txt", "w") as f:
    for i in indices:
        f.write(targets[i] + "\n")

# w=wave.open(infiles[0])

def concatenate_wav_files(input_files, output_file):
    combined = AudioSegment.empty()
    for file_name in input_files:
        sound = AudioSegment.from_wav(file_name)
        combined += sound
    combined.export(output_file, format="wav")

# # Example usage:
# input_files = ["input1.wav", "input2.wav", "input3.wav"]  # Add more file names as needed
# output_file = "output.wav"
concatenate_wav_files(infiles, outfile)
