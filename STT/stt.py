import torch
import torchaudio
from transformers import pipeline
from googletrans import Translator

def speech_to_text(waveform, sample_rate, translate_from, translate_to):
    # translate_from eg. 'en', 'ja'
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("device = ", device)

    pipe = pipeline(
        "automatic-speech-recognition", model="openai/whisper-medium", device=device
    )

    # waveform, sample_rate = torchaudio.load(audio_path)

    # Resampling
    target_sample_rate = 16000
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)

    # Convert into dictionary format
    custom_sample = {
        "audio": {
            "array": waveform.squeeze().numpy(),
            "sampling_rate": target_sample_rate
        },
        "id": "custom_sample_id"
    }


    # # Display custom audio using IPython
    # from IPython.display import Audio
    # Audio(data=waveform.squeeze().numpy(), rate=target_sample_rate)


    audio = waveform.reshape((-1,)).numpy()
    outputs = pipe(audio, max_new_tokens=256, generate_kwargs={"task": "translate"})

    audio_embeddings = outputs["text"]


    translator = Translator()
    text = audio_embeddings
    translated_text = translator.translate(text, src=translate_from, dest=translate_to)
    # print(translated_text.text)

    # f = open(text_path, "w")
    # f.write(translated_text.text)
    # f.close()
    return translated_text.text