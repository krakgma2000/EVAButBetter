from speech_recognition_wrapper import SpeechRecognitionSystem
import pywhisper as whisper
import numpy as np
import torch
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_AUDIO_FILE = os.path.join(os.path.dirname(os.getcwd()), os.path.join("data", "test.mp3"))
LANGUAGE = "en"

model_whisper = whisper.load_model("base", device=DEVICE)
options_whisper = whisper.DecodingOptions(without_timestamps=True)


class SpeechRecognitionWhisper(SpeechRecognitionSystem):
    def __init__(self, model=model_whisper, options=options_whisper):
        super().__init__(model)
        self.options = options

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

    def run(self, audio_file):
        audio = whisper.load_audio(audio_file)
        mel = self.to_mel(audio)
        predicted_lang = self.predict_language(mel)
        self.set_options(LANGUAGE, without_timestamps=True, fp16=False)
        decoded_audio = self.decode(mel)
        if decoded_audio.no_speech_prob > 0.5:
            return None
        else:
            return predicted_lang, decoded_audio.text

    def run_test(self, audio_file):
        audio = whisper.load_audio(audio_file)
        mel = self.to_mel(audio)
        predicted_lang = self.predict_language(mel)
        self.set_options("en", without_timestamps=True, fp16=False)
        decoded_audio = self.decode(mel)
        return predicted_lang, decoded_audio


speechRecSys = SpeechRecognitionWhisper(model_whisper)
speechRecSys.print_description()
testResult = speechRecSys.run_test(INPUT_AUDIO_FILE)
print(testResult)
