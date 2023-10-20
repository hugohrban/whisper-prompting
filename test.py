from faster_whisper import WhisperModel
from datasets import load_dataset



model_size = "large-v2"

# Run on GPU with FP16
model = WhisperModel(model_size, device="cuda", compute_type="float16")

# fleurs = load_dataset("google/fleurs", "cs_cz", split="test")

data = "en_fleurs/all_concatenated.wav"

segments, info = model.transcribe(data, beam_size=5)

# print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

for segment in segments:
    print(segment.text)
