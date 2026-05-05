from TTS.api import TTS

tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

tts.tts_to_file(
    text=text,
    speaker_wav="clean.wav",
    language="en",
    file_path="output.wav"
)