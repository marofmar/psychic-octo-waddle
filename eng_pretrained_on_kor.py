import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

PATH = "audio/file/path"

def process(PATH):
    audio, sampling_rate = librosa.load(PATH, sr = 16000)
    tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    input_values = tokenizer(audio, return_tensors='pt').input_values
    logits = model(input_values).logits
    prediction = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(prediction)[0]
    print(transcription)
    return



if __name__ == "__main__":
    process("/home/yjchung/STT/723_sample_kor.WAV")
