from pydub import AudioSegment

audio = AudioSegment.from_file("myaudio.wav")
audio = audio.set_frame_rate(22050).set_channels(1)

audio.export("clean.wav", format="wav")