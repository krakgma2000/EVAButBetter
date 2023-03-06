import pyaudio
import math
import struct
import wave
import time
import os

THRESHOLD = 10

SHORT_NORMALIZE = (1.0 / 32768.0)
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
S_WIDTH = 2

TIMEOUT_LENGTH = 2

RECORD_DIR = "./records"


class Recorder:

    def rms(self, frame):
        count = len(frame) / S_WIDTH
        format = "%dh" % count
        shorts = struct.unpack(format, frame)

        sum_squares = 0.0
        for sample in shorts:
            n = sample * SHORT_NORMALIZE
            sum_squares += n * n
        rms = math.pow(sum_squares / count, 0.5)

        return rms * 1000

    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=FORMAT,
                                  channels=CHANNELS,
                                  rate=RATE,
                                  input=True,
                                  output=True,
                                  frames_per_buffer=CHUNK)

    def record(self):
        print('Noise detected, recording beginning')
        rec = []
        current = time.time()
        end = time.time() + TIMEOUT_LENGTH

        while current <= end:

            data = self.stream.read(CHUNK)
            if self.rms(data) >= THRESHOLD: end = time.time() + TIMEOUT_LENGTH

            current = time.time()
            rec.append(data)
        return self.write(b''.join(rec))

    def write(self, recording):
        n_files = len(os.listdir(RECORD_DIR))

        filename = os.path.join(RECORD_DIR, '{}.wav'.format(n_files))

        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(recording)
        wf.close()
        print('Written to file: {}'.format(filename))
        return filename

    def listen(self):
        print('Listening...')
        input = self.stream.read(CHUNK)
        rms_val = self.rms(input)
        if rms_val > THRESHOLD:
            return self.record()
