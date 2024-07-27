import os
import torch
import numpy as np
import streamlit as st
import argparse
from typing import Dict, List
from io import BytesIO
from pathlib import Path
from pyannote.database.util import load_rttm
from pyannote.core import Annotation
from pydub import AudioSegment
from pydub.playback import play
from STT.Speaker_diarization import create_rttm
from STT.stt import speech_to_text
from TextToSpeech.main import clone_voice


aud_path = "/Users/hazarathayyayallanki/Documents/Projects/holovoice/audios/a1_small.wav"
rttm_path = "./rttms/a1_small.rttm"
sample_path = "./samples/"
reference_path = "/Users/hazarathayyayallanki/Documents/Projects/holovoice/samples"
text_path = "./texts/current.txt"
model_path = "tts_models/multilingual/multi-dataset/xtts_v2"
cloned_voice_path = "./output.wav"

source_lang = 'en'
dest_lang = 'hi'
rttm = load_rttm(rttm_path)

def as_dict_list(annotation: Annotation) -> Dict[str, List[Dict]]:
    result = {label: [] for label in annotation.labels()}
    for segment, track, label in annotation.itertracks(yield_label=True):
        result[label].append({
            "speaker": label,
            "start": segment.start,
            "end": segment.end,
            "duration": segment.duration,
            "track": track,
        })
    return result

def speaker_audios(rttm):
    visited = []
    speakers = {}
    for uri, annotation in rttm.items():
        data = as_dict_list(annotation)
        for speaker in data.values():
            for time_stamp in speaker:
                if time_stamp['speaker'] not in visited:
                    visited += [time_stamp['speaker']]
                    speakers[time_stamp['speaker']] = []
                temp = [time_stamp['start'], time_stamp['end']]
                speakers[time_stamp['speaker']].append(temp)
    return speakers

speakers = speaker_audios(rttm)
# print(speakers)

def sample_audios(aud_path, speakers, time_limit, sample_path):
    audio = AudioSegment.from_file(aud_path)
    time_limit_ms = time_limit * 1000  # Convert time limit to milliseconds

    for speaker, segments in speakers.items():
        time_count = 0
        temp_aud = AudioSegment.empty()
        for segment in segments:
            start_ms = segment[0] * 1000  # Convert start time to milliseconds
            end_ms = segment[1] * 1000    # Convert end time to milliseconds
            time_count += end_ms - start_ms
            segment_audio = audio[start_ms:end_ms]
            temp_aud += segment_audio
            print(f"Time count for {speaker}: {time_count} ms")
            if time_count > time_limit_ms:
                break

        os.makedirs(sample_path, exist_ok=True)

        store_audio_path = os.path.join(sample_path, f"{speaker}.wav")
        temp_aud.export(store_audio_path, format="wav")
        print(f"Exported {store_audio_path}")

# This stores the reference audios for each speaker
# sample_audios(aud_path=aud_path, speakers=speakers, time_limit=6, sample_path=sample_path)

def get_sorted_items(rttm):
    all_segments = []
    for uri, annotation in rttm.items():
        data = as_dict_list(annotation)
        for speaker, segments in data.items():
            for segment in segments:
                all_segments.append(segment)
    return sorted(all_segments, key = lambda x:x['track'])

# sorted_segments = get_sorted_items(speakers)
# print(sorted_segments)


def audiosegment_to_waveform(audio_segment):
    raw_data = audio_segment.raw_data
    
    # Convert bytes to numpy array
    waveform = np.frombuffer(raw_data, dtype=np.int16)
    
    # Reshape numpy array to match the number of channels
    waveform = waveform.reshape(-1, audio_segment.channels).T
    
    # Convert numpy array to tensor
    waveform = torch.tensor(waveform, dtype=torch.float32)
    
    # Normalize waveform
    waveform /= 32768.0
    
    # Get sample rate
    sample_rate = audio_segment.frame_rate
    
    return waveform, sample_rate

def main(audio, source_lang, target_lang):
    wave_form, samp_rate = audiosegment_to_waveform(audio)
    rttm = create_rttm(wave_form, samp_rate, dest_path=rttm_path) # uncomment me
    # rttm = load_rttm(rttm_path) # comment me 
    speakers = speaker_audios(rttm)
    sorted_segments = get_sorted_items(rttm)

    # audio = AudioSegment.from_file(audio_path)
    final_audio = AudioSegment.empty()
    for segment in sorted_segments:
        start = segment['start']*1000
        end = segment['end']*1000
        temp_audio = audio[start:end]
        waveform, sample_rate = audiosegment_to_waveform(temp_audio)
        translated_text = speech_to_text(waveform=waveform, sample_rate=sample_rate, translate_from=source_lang, translate_to=target_lang)
        ref_path = reference_path + f"/{segment['speaker']}.wav"
        temp_cloned_audio = clone_voice(model_path=model_path, reference_path=ref_path, destination_path=cloned_voice_path, dest_lang=target_lang,  text=translated_text)
        final_audio += temp_cloned_audio
    return final_audio

# final_audio = main(aud_path, source_lang, dest_lang, sorted_segments)
# play(final_audio)
# final_audio.export("./final_output", format="wav")

languages_dict = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Polish": "pl",
    "Turkish": "tr",
    "Russian": "ru",
    "Dutch": "nl",
    "Czech": "cs",
    "Arabic": "ar",
    "Chinese": "zh-cn",
    "Japanese": "ja",
    "Hungarian": "hu",
    "Korean": "ko"
}




st.title("Holovoice")

# File uploader
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:
    input_audio = AudioSegment.from_file(uploaded_file)

    channels = input_audio.split_to_mono()
    input_audio = channels[0]

    languages = list(languages_dict.keys())
    input_language = st.selectbox("Select Input Audio Language", languages)
    target_language = st.selectbox("Select Target Audio Language", languages)

    input_language_code = languages_dict[input_language]
    target_language_code = languages_dict[target_language]
    st.write(f"Input Language: {input_language} ({input_language_code})")
    st.write(f"Target Language: {target_language} ({target_language_code})")

    if st.button("Translating"):
        st.write("Translating and cloning...")
        output_audio = main(input_audio, input_language_code, target_language_code)
        
        output_buffer = BytesIO()
        output_audio.export(output_buffer, format="wav")
        output_buffer.seek(0)
        
        st.audio(output_buffer, format='audio/wav')
        
        st.download_button(label="Download Processed Audio",
                        data=output_buffer,
                        file_name="processed_audio.wav",
                        mime="audio/wav")


