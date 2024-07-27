from TTS.api import TTS
from pathlib import Path
from pydub import AudioSegment

def clone_voice(model_path, reference_path, destination_path, dest_lang, text):
    tts = TTS(model_path, gpu=False)

    # # generate speech by cloning a voice using default settings
    # f = open("./texts/wh_medium_ja.txt", "r")
    # text = f.read()
    # f.close()
    # # print(text)

    tts.tts_to_file(text=text,
                    file_path=destination_path, # it was wav
                    speaker_wav=[reference_path], # it was mp3
                    language=dest_lang,
                    split_sentences=True
                    )
    return AudioSegment.from_file(destination_path)
    