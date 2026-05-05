from TTS.api import TTS

tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")

tts.tts_to_file(
    text="Hello Ravi, this is your cloned voice working successfully",
    speaker_wav="clean.wav",
    language="en",
    file_path="output.wav"
)

print("DONE")