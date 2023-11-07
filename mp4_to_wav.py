from moviepy.editor import VideoFileClip
import sys
import os
import pandas as pd
def convert_to_wav(input_file, output_file):
    video = VideoFileClip(input_file)
    audio = video.audio
    audio.write_audiofile(output_file)

if __name__ == "__main__":
    input_prefix = "/Users/hugohrban/Documents/random_scripts/whisper_stuff/ESIC-v1.0/v1.0/dev/"
    # for every folder in the dev folder
    for folder in os.listdir(input_prefix):
        # for every file in the folder
        if os.path.isdir(input_prefix + folder):
            for subfolder in os.listdir(input_prefix + folder):
                # if the file is a video file
                if os.path.isdir(input_prefix + folder + "/" + subfolder):
                    for file in os.listdir(input_prefix + folder + "/" + subfolder):
                        # if file == "en.IStde.man.vert+ts":
                        if file.endswith(".mp4"):
                            # convert it to wav
                            input_file = input_prefix + folder + "/" + subfolder + "/" + file
                            output_file = input_prefix + folder + "/" + subfolder + "/" + "audio.wav"
                            print(f"Converting {input_file} to {output_file}")
                            convert_to_wav(input_file, output_file)

               
    
    # input_file = "/Users/hugohrban/Documents/random_scripts/whisper_stuff/ESIC-v1.0/v1.0/dev/20080901/018_006_EN_Ždanoka/en.OS.man-diar.mp4"
    # output_file = "/Users/hugohrban/Documents/random_scripts/whisper_stuff/ESIC-v1.0/v1.0/dev/20080901/018_006_EN_Ždanoka/audio.wav"

    # convert_to_wav(input_file, output_file)
