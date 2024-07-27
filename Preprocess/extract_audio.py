from moviepy.editor import VideoFileClip

def extract_audio(video_file, audio_output):
    clip = VideoFileClip(video_file)
    audio = clip.audio
    audio.write_audiofile(audio_output)
    clip.close()


video_file = "./videos/theshawshankredemption.mp4"
audio_output = "./audios/a1.wav"
extract_audio(video_file, audio_output)
