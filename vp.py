import numpy as np
import librosa
import scipy.signal
import os
import sounddevice as sd
import soundfile as sf
import sys


# commands
# 1. open calculator
# 2. open notepad
# 3. open browser
# 4. on light
# 5. off light
# 6. custom command

def record_audio(filename):
    print("Recording...")
    
    samplerate = 16000  # prefer for voice commands / speech recognition
    channels = 1
    out_file = filename

    with sf.SoundFile(out_file, mode='w', samplerate=samplerate, channels=channels, subtype='PCM_16') as file:
        def callback(indata, frames, time, status):
            if status:
                print(status, file=sys.stderr)
            file.write(indata)
        with sd.InputStream(samplerate=samplerate, channels=channels, callback=callback):
            input("Recording... press Enter to stop\n")
    print(f"Saved to {out_file}")