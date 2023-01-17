# This is a sample Python script.

# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import whisper
import numpy as np


class SpeechRecognitionSystem:
    def __init__(self, model):
        self.model = model
        self.options = whisper.DecodingOptions(without_timestamps=True)

    def print_description(self):
        print(
            f"Model is {'multilingual' if self.model.is_multilingual else 'English-only'} "
            f"and has {sum(np.prod(p.shape) for p in self.model.parameters()):,} parameters."
        )

    def decode(self, audio):
        return whisper.decode(self.model, audio, self.options)

    def predict_language(self, audio):
        _, probs = self.model.detect_language(audio)
        return max(probs, key=probs.get)

    def set_options(self, language, without_timestamps):
        self.options = whisper.DecodingOptions(language=language, without_timestamps=without_timestamps)
