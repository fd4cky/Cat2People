import os, shutil
import matplotlib.pyplot as plt
from tensorflow import audio as tfAudio, io, signal, squeeze, abs
from keras import layers, Model, Input, callbacks
from keras.utils import image_dataset_from_directory

def sort_dataset(path):
    folder_name = 'data'
    files = os.listdir(path=path)
    labels = ['Brushing', 'Food', 'Isolation']
    os.mkdir(folder_name)
    for label in labels:
        os.mkdir(f"{folder_name}/{label}")
    for file in files:
        if file[0] == "B":
            shutil.move(f"{path}/{file}", f"{folder_name}/Brushing")
        elif file[0] == "F":
            shutil.move(f"{path}/{file}", f"{folder_name}/Food")
        else:
            shutil.move(f"{path}/{file}", f"{folder_name}/Isolation")

def load_audio(path):
    audio_binary = io.read_file(path)
    audio, _ = tfAudio.decode_wav(audio_binary)
    waveform = squeeze(audio, axis=-1)
    return waveform

def load_audio_files(path, label):
    result = []
    files = io.gfile.glob(f'{path}/{label}/*')
    for file in files:
        result.append(load_audio(file))
    return result

def plot_audio(data):
    plt.figure()
    plt.plot(data[0].numpy())
    plt.title(data[1])
    plt.show()

def get_spectrogram(waveform): 
    spectrogram = signal.stft(waveform, frame_length=255, frame_step=128)
    spectrogram = abs(spectrogram)
    return spectrogram

def create_data_images(path, val_split):
    os.mkdir('data_images')
    os.mkdir('data_images/train')
    os.mkdir('data_images/validation')

    for label in ['Brushing', 'Food', 'Isolation']:
        os.mkdir(f'data_images/train/{label}')
        os.mkdir(f'data_images/validation/{label}')
        files = load_audio_files(path, label)
        files_count = len(files)
        for i, file in enumerate(files):
            spectrogram = get_spectrogram(file)
            if i <= (files_count * (1 - val_split)):
                plt.imsave(f"data_images/train/{label}/spec_img{i}.png", spectrogram.numpy(), cmap='gray')
            else:
                plt.imsave(f"data_images/validation/{label}/spec_img{i}.png", spectrogram.numpy(), cmap='gray')

#create_data_images('data', val_split=0.2)
img_size = (180, 180)
batch_size = 32
epochs = 16
train_data = image_dataset_from_directory('data_images/train', image_size=img_size, batch_size=batch_size)
val_data = image_dataset_from_directory('data_images/validation', image_size=img_size, batch_size=batch_size)
num_classes = len(train_data.class_names)

def get_model(img_size, num_classes):
    inputs = Input(shape=(img_size + (3,)))
    x = layers.Rescaling(1.0/255)(inputs)
    x = layers.Conv2D(8, 2, strides=2, activation='relu', padding='same')(x)
    x = layers.Conv2D(8, 2, activation='relu', padding='same')(x)
    x = layers.Conv2D(16, 2, strides=2, activation='relu', padding='same')(x)
    x = layers.Conv2D(16, 2, activation='relu', padding='same')(x)
    x = layers.Conv2D(32, 2, strides=2, activation='relu', padding='same')(x)
    x = layers.Conv2D(32, 2, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=3)(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    return model

model = get_model(img_size, num_classes)
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

callback = [callbacks.ModelCheckpoint('audio_model2.keras', save_best_only=True)]

history = model.fit(train_data, validation_data=val_data, epochs=epochs, callbacks=callback, batch_size=batch_size)

val_accuracy = model.history.history["val_accuracy"]
val_loss = model.history.history["val_loss"]
plt.plot(val_accuracy, label="val_accuracy")
plt.plot(val_loss, label="val_loss")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend()
plt.show()