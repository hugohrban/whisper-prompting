from moviepy.editor import VideoFileClip
import sys
import os
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm

def get_sentence_timestamps(file_name):
    # input_file = input_prefix + folder + "/" + subfolder + "/" + file
    sentence = 1
    timestamps = [0]
    with open (file_name, "r") as f:
        for line in f:
            tokens = line.split("\t")
            start_time = tokens[0]
            end_time = tokens[1]
            word = tokens[2]
            sentence_num= int(tokens[4])
            if sentence_num > sentence:
                sentence = sentence_num
                timestamps.append(start_time)
        timestamps.append(end_time)
    return list(map(float, timestamps))

def split_audio_by_timestamps(input_wav, output_prefix, timestamps_seconds):
    audio, sr = librosa.load(input_wav, sr=None)
    for i, timestamp in enumerate(timestamps_seconds):
        start_sample = int(timestamp * sr)
        end_sample = int(timestamps_seconds[i + 1] * sr) if i + 1 < len(timestamps_seconds) else None

        if end_sample:
            chunk = audio[start_sample:end_sample]
        else:
            chunk = audio[start_sample:]

        output_path = f"{output_prefix}_{i+1}.wav"
        sf.write(output_path, chunk, sr)

def split_audio(audio_fn, timestamps):
	print('ts:', len(timestamps), timestamps)
	for i in range(0, len(timestamps)-1):
		beg = timestamps[i]
		end = timestamps[i+1]

		a, _ = librosa.load(audio_fn, sr=16000)
		beg_s = int(beg*16000)
		end_s = int(end*16000)
		splitted = a[beg_s:end_s]
		new_fn = audio_fn.replace('.wav', f'-{i+1}.wav')
		sf.write(new_fn, splitted, 16000)

def split_all_audio(d):
    file = d + "/en.OSt.man.vert+ts"
    audio = d + "/audio.wav"
    split_audio(audio, get_sentence_timestamps(file))

def process_all():
    input_prefix = "/Users/karelvlk/Developer/mff/ufal/whisper-prompting/ESICv1.0/dev/"
    i = 0
    # for every folder in the dev folder
    for folder in os.listdir(input_prefix):
        # for every file in the folder
        if os.path.isdir(input_prefix + folder):
            for subfolder in os.listdir(input_prefix + folder):
                # if the file is a video file
                if os.path.isdir(input_prefix + folder + "/" + subfolder):
                    for file in os.listdir(input_prefix + folder + "/" + subfolder):
                        if file == "en.IStde.man.vert+ts":
                            input_file_d = input_prefix + folder + "/" + subfolder
                            split_all_audio(input_file_d)
                            i += 1
                            print('done', i)

process_all()
