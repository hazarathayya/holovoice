from moviepy.editor import VideoFileClip

def extract_audio(video_file, audio_output):
    clip = VideoFileClip(video_file)
    audio = clip.audio
    audio.write_audiofile(audio_output)
    clip.close()


video_file = "input_video.mp4"
audio_output = "output_audio.wav"
extract_audio(video_file, audio_output)
