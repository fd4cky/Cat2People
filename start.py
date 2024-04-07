import numpy as np
import matplotlib.pyplot as plt
from tensorflow import audio as tfAudio, signal, abs, expand_dims, io, squeeze
from keras.utils import img_to_array, load_img
from keras.models import load_model
from subprocess import call

modelPath = r'audio_model.keras'
img_size = (180, 180)
labels = ['Brushing', 'Food', 'Isolation']
description = {'Brushing':'Кот чистит себя.', 'Food':'Кот хочет есть.', 'Isolation':'Кот в незнакомой ему обстановке.'}
model = load_model(modelPath)

def load_audio(path):
    audio_binary = io.read_file(path)
    audio, _ = tfAudio.decode_wav(audio_binary)
    waveform = squeeze(audio, axis=-1)
    return waveform

def get_spectrogram(waveform): 
    spectrogram = signal.stft(waveform, frame_length=255, frame_step=128)
    spectrogram = abs(spectrogram)
    return spectrogram

def recognition(audio):
    path = audio
    plt.imsave('testimg/test.png', get_spectrogram(load_audio(path)).numpy(), cmap='gray')
    img = load_img('testimg/test.png', target_size=(img_size))
    img_array = img_to_array(img)
    img_array = expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    
    #debug return
    #return f'model: {modelPath.replace(".keras", "")}\n{predictions[0]}\n{labels}\n\n{labels[np.argmax(predictions[0])]} {round(max(predictions[0])*100, 1)}%\nDescription: {description[labels[np.argmax(predictions[0])]]}'
    return f'{labels[np.argmax(predictions[0])]} {round(max(predictions[0])*100, 1)}%\nDescription: {description[labels[np.argmax(predictions[0])]]}'

def convert_ogg_to_wav(path):
    call(['ffmpeg', '-i', f'{path}.ogg', '-acodec', 'pcm_s16le', '-ac', '1', '-ar', '16000', f'{path}.wav'])