#!/bin/bash

# # czech version
# if [ ! -d "cs_fleurs" ]; then
#     mkdir cs_fleurs
# fi

# for i in {0..99}
# do
#     curl --output "cs_fleurs/audio_$i.wav" https://datasets-server.huggingface.co/assets/google/xtreme_s/--/fleurs.cs_cz/test/$i/audio/audio.wav
# done

# english 
if [ ! -d "en_fleurs" ]; then
    mkdir en_fleurs
fi
for i in {0..99}
do
    curl --output "en_fleurs/audio_$i.wav" https://datasets-server.huggingface.co/assets/google/xtreme_s/--/fleurs.en_us/test/$i/audio/audio.wav
done