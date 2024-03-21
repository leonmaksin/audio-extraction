import sounddevice as sd
import numpy as np
import librosa
from scipy.signal import find_peaks
import pandas as pd
from math import pi
import matplotlib.pyplot as plt
import threading
import queue

samplerate = 44100
windowms = 100
window = samplerate * windowms / 1000
sliderate = 0.3
seconds = 0
update_queue = queue.Queue()

# Define a buffer to accumulate audio frames (needed for some features that require a larger context)
buffer = np.array([])

# define our chart
categories=librosa.hz_to_note(440.0 * (2.0 ** np.linspace(0, 1, 13)), octave=False)
categories=categories[:-1]
N = len(categories)
angles = [n / float(N) * 2 * pi for n in range(N)]
ax = plt.subplot(111, polar=True)

def top_k_pitches(k):
    global buffer, samplerate

    # Perform FFT and get the magnitude spectrum
    fft_result = np.fft.rfft(buffer)
    magnitude_spectrum = np.abs(fft_result)
    freqs = np.fft.rfftfreq(len(buffer), 1/samplerate)
    peaks, _ = find_peaks(magnitude_spectrum, height=0.01 * np.max(magnitude_spectrum))

    k = min(k, len(peaks))

    top_k_indices = np.flip(np.argsort(magnitude_spectrum[peaks])[-k:])
    top_k_freqs = freqs[peaks][top_k_indices]
    top_k_magnitudes = magnitude_spectrum[peaks][top_k_indices]
    top_k_notes = librosa.hz_to_note(top_k_freqs, octave=False)

    note_mags = { k:0 for k in librosa.hz_to_note(440.0 * (2.0 ** np.linspace(0, 1, 13)), octave=False)}
    for idx in range(k):
        note_mags[top_k_notes[idx]] += top_k_magnitudes[idx]

    update_queue.put(note_mags)

    return top_k_freqs

def draw_chart(note_mags):
    values = [note_mags[cat] for cat in categories]

    # plt.cla()
    ax.clear()

    # Draw one axe per variable + add labels
    plt.xticks(angles, categories, color='grey', size=8)
    plt.ylim(0,100)

    # Plot data
    ax.plot(angles, values, linewidth=1, linestyle='solid')
    
    # Fill area
    ax.fill(angles, values, 'b', alpha=0.1)

    # Show the graph
    plt.pause(0.01)

# This callback function will be called by sounddevice for each new audio block
def audio_callback(indata, frames, time, status):
    global buffer
    global seconds
    if status:
        print(status)
    
    # append data to our buffer
    audio_data = indata[:, 0]
    buffer = np.concatenate((buffer, audio_data))
    seconds += frames / samplerate

    if len(buffer) >  2 * samplerate:
        buffer = buffer[-2 * samplerate:]
    
    # Check if buffer has enough data to process (e.g., 2048 samples)
    if len(buffer) >= window:
        print(f"============================ {seconds} seconds ============================")

        # Pitch
        top_pitches = top_k_pitches(100)

        # Move the buffer after processing
        buffer = buffer[-int(len(buffer) * sliderate):]

# Set the audio stream parameters
stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=samplerate)

# Start the audio stream asynchronously to process incoming audio in real-time
# with stream:
#     sd.sleep(60000)  # Keep the stream alive for 60 seconds

def start_audio_stream():
    with stream:
        sd.sleep(60000)  # Keep the stream alive for the desired duration

# Start the audio stream in a background thread
audio_thread = threading.Thread(target=start_audio_stream)
audio_thread.start()

with stream:
    try:
        while True:
            if not update_queue.empty():
                note_mags = update_queue.get_nowait()  # Get data from the queue
                draw_chart(note_mags)  # Update the plot with the new data
    except KeyboardInterrupt:
        print("Stopping...")