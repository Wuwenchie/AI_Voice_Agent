import struct
from pvrecorder import PvRecorder
import keyboard
import wave

recorder = PvRecorder(device_index=-1, frame_length=512)
audio = []

try:
    recorder.start()
    print("Recording... Press Ctrl+C to stop.")

    while True:
        frame = recorder.read()
        audio.extend(frame)

except KeyboardInterrupt:
    recorder.stop()
    with wave.open('test.wav', 'w') as f:
        f.setparams((1, 2, 16000, 512, "NONE", "NONE"))
        f.writeframes(struct.pack("h" * len(audio), *audio))
        AudioSegment.from_wav("test.wav").export("test.mp3", format="mp3")
    print("Audio saved to test.mp3")

finally:
    recorder.delete()
