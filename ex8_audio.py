import os 
import pathlib 
import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns 
import tensorflow as tf 
from tensorflow.keras import layers 
from tensorflow.keras import models 
from IPython import display 
# Set the seed value for experiment reproducibility. 
seed = 42 
tf.random.set_seed(seed) 
np.random.seed(seed
DATASET_PATH = 'data/mini_speech_commands' 
data_dir = pathlib.Path(DATASET_PATH) 
if not data_dir.exists(): 
tf.keras.utils.get_file( 
'mini_speech_commands.zip', 
origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip", 
extract=True, 
cache_dir='.', cache_subdir='data') 
plt.figure(figsize=(16, 10)) 
rows = 3 
cols = 3 
n = rows * cols 
for i in range(n): 
plt.subplot(rows, cols, i+1) 
audio_signal = example_audio[i] 
plt.plot(audio_signal) 
plt.title(label_names[example_labels[i]]) 
plt.yticks(np.arange(-1.2, 1.2, 0.2)) 
plt.ylim([-1.1, 1.1]) 
def get_spectrogram(waveform): 
# Convert the waveform to a spectrogram via a STFT. 
spectrogram = tf.signal.stft( 
waveform, frame_length=255, frame_step=128) 
# Obtain the magnitude of the STFT. 
spectrogram = tf.abs(spectrogram) 
# Add a `channels` dimension, so that the spectrogram can be used 
# as image-like input data with convolution layers (which expect 
# shape (`batch_size`, `height`, `width`, `channels`). 
spectrogram = spectrogram[..., tf.newaxis] 
return spectrogram
def plot_spectrogram(spectrogram, ax): 
if len(spectrogram.shape) > 2: 
assert len(spectrogram.shape) == 3 
spectrogram = np.squeeze(spectrogram, axis=-1) 
# Convert the frequencies to log scale and transpose, so that the time is 
# represented on the x-axis (columns). 
# Add an epsilon to avoid taking a log of zero. 
log_spec = np.log(spectrogram.T + np.finfo(float).eps) 
height = log_spec.shape[0] 
width = log_spec.shape[1] 
X = np.linspace(0, np.size(spectrogram), num=width, dtype=int) 
Y = range(height) 
ax.pcolormesh(X, Y, log_spec) 
rows = 3 
cols = 3 
n = rows*cols 
fig, axes = plt.subplots(rows, cols, figsize=(16, 9)) 
for i in range(n): 
r = i // cols 
c = i % cols 
ax = axes[r][c] 
plot_spectrogram(example_spectrograms[i].numpy(), ax) 
ax.set_title(label_names[example_spect_labels[i].numpy()]) 
plt.show()
model = models.Sequential([ 
layers.Input(shape=input_shape), 
# Downsample the input. 
layers.Resizing(32, 32), 
# Normalize. 
norm_layer, 
layers.Conv2D(32, 3, activation='relu'), 
layers.Conv2D(64, 3, activation='relu'), 
layers.MaxPooling2D(), 
layers.Dropout(0.25), 
layers.Flatten(), 
layers.Dense(128, activation='relu'), 
layers.Dropout(0.5), 
layers.Dense(num_labels), 
]) 
model.summary() 
model.compile( 
optimizer=tf.keras.optimizers.Adam(), 
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
metrics=['accuracy'], 
) 
EPOCHS = 10 
history = model.fit( 
train_spectrogram_ds, 
validation_data=val_spectrogram_ds, 
epochs=EPOCHS, 
callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2), 
)
