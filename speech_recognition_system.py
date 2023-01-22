import pywhisper as whisper
import numpy as np
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_AUDIO_FILE = "input.mp3"
LANGUAGE = "en"


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

    def predict_language(self, mel):
        _, probs = self.model.detect_language(mel)
        return max(probs, key=probs.get)

    def set_options(self, language, without_timestamps, fp16):
        self.options = whisper.DecodingOptions(language=language, without_timestamps=without_timestamps, fp16=fp16)

    def to_mel(self, audio):
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
        return mel

    def run(self):
        audio = whisper.load_audio(INPUT_AUDIO_FILE)
        mel = self.to_mel(audio)
        predicted_lang = self.predict_language(mel)
        self.set_options(LANGUAGE, without_timestamps=True, fp16=False)
        decoded_audio = self.decode(mel)
        if decoded_audio.no_speech_prob > 0.5:
            return None
        else:
            return predicted_lang, decoded_audio.text

    def run_test(self):
        audio = whisper.load_audio("test.mp3")
        mel = self.to_mel(audio)
        predicted_lang = self.predict_language(mel)
        self.set_options("en", without_timestamps=True, fp16=False)
        decoded_audio = self.decode(mel)
        return predicted_lang, decoded_audio


speechRecSys = SpeechRecognitionSystem(whisper.load_model("base", device=DEVICE))
speechRecSys.print_description()
testResult = speechRecSys.run_test()
print(testResult)

