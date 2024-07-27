import streamlit as st
from pydub import AudioSegment
from io import BytesIO


# Streamlit app
st.title("Audio Processing App")

# File uploader
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:
    # Read the uploaded file as an AudioSegment
    input_audio = AudioSegment.from_file(uploaded_file)
    
    # Call the main processing function
    output_audio = main(input_audio)
    
    # Export the processed audio to a BytesIO object
    output_buffer = BytesIO()
    output_audio.export(output_buffer, format="wav")
    output_buffer.seek(0)
    
    # Display audio player for the processed audio
    st.audio(output_buffer, format='audio/wav')
    
    # Provide download link for the processed audio
    st.download_button(label="Download Processed Audio",
                       data=output_buffer,
                       file_name="processed_audio.wav",
                       mime="audio/wav")
