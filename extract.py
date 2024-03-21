import eventlet
eventlet.monkey_patch()

import sounddevice as sd
import numpy as np
import librosa
from scipy.signal import find_peaks
from flask import Flask
from flask_socketio import SocketIO
import queue

samplerate = 44100
windowms = 100
window = samplerate * windowms / 1000
sliderate = 0.3
seconds = 0
data_queue = queue.Queue()

# Define a buffer to accumulate audio frames (needed for some features that require a larger context)
buffer = np.array([])
buffer_long = np.array([])

def top_k_pitches(k):
    global buffer, samplerate

    # Perform FFT and get the magnitude spectrum
    fft_result = np.fft.rfft(buffer)
    magnitude_spectrum = np.abs(fft_result)
    freqs = np.fft.rfftfreq(len(buffer), 1/samplerate)
    peaks, _ = find_peaks(magnitude_spectrum, height=0.01 * np.max(magnitude_spectrum))

    top_k_indices = np.flip(np.argsort(magnitude_spectrum[peaks])[-k:])
    top_k_freqs = freqs[peaks][top_k_indices]
    top_k_magnitudes = magnitude_spectrum[peaks][top_k_indices]
    top_k_notes = librosa.hz_to_note(top_k_freqs, octave=False)

    note_mags = { k:0 for k in librosa.hz_to_note(440.0 * (2.0 ** np.linspace(0, 1, 13)), octave=False)}
    for idx in range(k):
        note_mags[top_k_notes[idx]] += top_k_magnitudes[idx]

    return top_k_freqs

# This callback function will be called by sounddevice for each new audio block
def audio_callback(indata, frames, time, status):
    global buffer
    global buffer_long
    global seconds
    if status:
        print(status)
    
    # append data to our buffer
    audio_data = indata[:, 0]
    buffer = np.concatenate((buffer, audio_data))
    buffer_long = np.concatenate((buffer_long, audio_data))
    seconds += frames / samplerate

    if len(buffer) >  2 * samplerate:
        buffer = buffer[-2 * samplerate:]
    
    # Check if buffer has enough data to process (e.g., 2048 samples)
    if len(buffer) >= window:
        print(f"============================ {seconds} seconds ============================")

        # Calculate RMS energy
        rms_energy = np.sqrt(np.mean(buffer**2))
        print(f"RMS Energy: {rms_energy:.3f}") # loudness

        # Spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=buffer, sr=samplerate)[0, 0]
        print(f"Spectral Centroid: {spectral_centroid:.2f} Hz") # average frequency, brightness of the sound

        # Spectral flatness
        spectral_flatness = librosa.feature.spectral_flatness(y=buffer)[0, 0]
        print(f"Spectral Flatness: {spectral_flatness:.3f}") # how "noisy" vs "tone-like" is the sound?

        # Spectral roll-off
        spectral_rolloff = librosa.feature.spectral_rolloff(y=buffer, sr=samplerate)[0, 0]
        print(f"Spectral Rolloff: {spectral_rolloff:.2f} Hz") # "densest" frequency, how bright the highest contributing sound is

        # Pitch
        top_pitches = top_k_pitches(10)
        print(f"Top 10 Pitches: {top_pitches}")

        # MFCCs (first 13 coefficients)
        mfccs = librosa.feature.mfcc(y=buffer, sr=samplerate, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)
        print(f"MFCCs (mean): {mfccs_mean}") # best suited for ML tasks. no good interpretation

        # beat tracking
        tempo, beats = librosa.beat.beat_track(y=buffer_long, sr=samplerate)

        # 'tempo' is the estimated tempo in beats per minute (BPM)
        # 'beats' are the frame indices of detected beats
        print(f"Tempo: {tempo:.2f} BPM")
        print(f"Beats: {beats}")

        data_queue.put({'top_pitch': top_pitches[0]})

        # Move the buffer after processing
        buffer = buffer[-int(len(buffer) * sliderate):]

# Set the audio stream parameters
stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=samplerate)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

@socketio.on('ready')
def handle_ready():
    print("Client is ready")
    # # # Start the audio stream asynchronously to process incoming audio in real-time
    stream.start()
    try:
        while True:
            if not data_queue.empty():
                data = data_queue.get_nowait()  # Get data from the queue
                socketio.emit('data', data)
                eventlet.sleep(0.01)
    except KeyboardInterrupt:
        print("Stopping...")
        stream.stop()

if __name__ == '__main__':
    socketio.run(app, port=5001)

