import sounddevice as sd
import numpy as np
import librosa

samplerate = 44100
windowms = 100
window = samplerate * windowms / 1000
sliderate = 0.3
seconds = 0

# Define a buffer to accumulate audio frames (needed for some features that require a larger context)
buffer = np.array([])
buffer_long = np.array([])

def top_k_pitches(k):
    pitches, magnitudes = librosa.piptrack(y=buffer, sr=samplerate, fmin=40.0)

    top_magnitude_indices = np.argsort(magnitudes.flatten())[-k:]

    # Initialize an array to hold the top k pitch values
    top_pitches = []

    # Loop through the indices of the top k magnitudes
    for index in top_magnitude_indices:
        # Convert the 1D index back to 2D indices for pitches and magnitudes arrays
        mag_index_row = index % pitches.shape[0]
        mag_index_col = index // pitches.shape[0]

        # Get the pitch value corresponding to the current magnitude index
        pitch = pitches[mag_index_row, mag_index_col]

        # Check if the pitch value is positive (indicating a detected pitch)
        if pitch > 0:
            top_pitches.append(pitch)

    return top_pitches

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
        print(f"RMS Energy: {rms_energy:.3f}")

        # Spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=buffer, sr=samplerate)[0, 0]
        print(f"Spectral Centroid: {spectral_centroid:.2f} Hz")

        # Spectral flatness
        spectral_flatness = librosa.feature.spectral_flatness(y=buffer)[0, 0]
        print(f"Spectral Flatness: {spectral_flatness:.3f}")

        # Spectral roll-off
        spectral_rolloff = librosa.feature.spectral_rolloff(y=buffer, sr=samplerate)[0, 0]
        print(f"Spectral Rolloff: {spectral_rolloff:.2f} Hz")

        # Pitch
        top_pitches = top_k_pitches(10)
        if top_pitches:
            print(f"Top 10 Pitches: {top_pitches}")
        else:
            print("No pitches detected")

        # MFCCs (first 13 coefficients)
        mfccs = librosa.feature.mfcc(y=buffer, sr=samplerate, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)
        print(f"MFCCs (mean): {mfccs_mean}")

        # beat tracking
        tempo, beats = librosa.beat.beat_track(y=buffer_long, sr=samplerate)

        # 'tempo' is the estimated tempo in beats per minute (BPM)
        # 'beats' are the frame indices of detected beats
        print(f"Tempo: {tempo:.2f} BPM")
        print(f"Beats: {beats}")

        # Move the buffer after processing
        buffer = buffer[-int(len(buffer) * sliderate):]

# Set the audio stream parameters
stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=samplerate)

# Start the audio stream asynchronously to process incoming audio in real-time
with stream:
    sd.sleep(60000)  # Keep the stream alive for 60 seconds
