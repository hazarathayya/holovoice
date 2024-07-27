# Read audio
data, samplerate = sf.read('Vocals.wav')
# reduce noise
y_reduced_noise = nr.reduce_noise(y=data, sr=samplerate)
# save audio
sf.write('Vocals_reduced.wav', y_reduced_noise, samplerate, subtype="PCM_24")
# load and play audio
data, samplerate = librosa.load('Vocals_reduced.wav')
ipd.Audio('Vocals_reduced.wav')
