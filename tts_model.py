# from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
# from datasets import load_dataset
# import torch
# import soundfile as sf
# import tempfile
# from io import BytesIO

# # Cache everything when imported via Streamlit
# def load_tts_models():
#     processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
#     model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
#     vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
#     embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
#     speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
#     return processor, model, vocoder, speaker_embeddings

# def synthesize_speech(text):
#     processor, model, vocoder, speaker_embeddings = load_tts_models()
#     inputs = processor(text=text, return_tensors="pt")
#     speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

#     # Save to temp WAV file
#     # with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
#     #     sf.write(f.name, speech.numpy(), samplerate=16000)
#     #     return f.name  # Return file path
    
#     buffer = BytesIO()
#     sf.write(buffer, speech.numpy(), samplerate=16000, format='WAV')
#     buffer.seek(0)
#     return buffer

# tts_model.py
from functools import lru_cache
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch, soundfile as sf
from io import BytesIO

@lru_cache(maxsize=1)
def _load_bundle():
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
    xvec = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    spk = torch.tensor(xvec[7306]["xvector"]).unsqueeze(0)
    return processor, model, vocoder, spk

def synthesize_speech(text: str) -> BytesIO:
    processor, model, vocoder, spk = _load_bundle()
    ids = processor(text=text, return_tensors="pt")["input_ids"]
    wav = model.generate_speech(ids, spk, vocoder=vocoder)
    buf = BytesIO()
    sf.write(buf, wav.numpy(), samplerate=16000, format="WAV")
    buf.seek(0)
    return buf
