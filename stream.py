import pyaudio
import wave
import numpy as np
import time
import matplotlib.pyplot as plt

#AUDIO INPUT
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 2
WAVE_OUTPUT_FILENAME = "output.wav"

audio = pyaudio.PyAudio()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(audio.get_sample_size(FORMAT))
wf.setframerate(RATE)

f,ax = plt.subplots(2)
x = np.arange(10000)
y = np.random.randn(10000)

# Plot 0 is for raw audio data
plot_raw, = ax[0].plot(x, y)
ax[0].set_xlim(0,1000)
ax[0].set_ylim(-5000,5000)
ax[0].set_title("Raw Audio Signal")
# Plot 1 is for the FFT of the audio
plot_fft, = ax[1].plot(x, y)
ax[1].set_xlim(0,5000)
ax[1].set_ylim(-100,100)
ax[1].set_title("FFT")

def plot_data(in_data):
    audio_data = np.fromstring(in_data, np.int16)
    dfft = 10.*np.log10(abs(np.fft.rfft(audio_data)))

    plot_raw.set_xdata(np.arange(len(audio_data)))
    plot_raw.set_ydata(audio_data)
    plot_fft.set_xdata(np.arange(len(dfft))*10.)
    plot_fft.set_ydata(dfft)

    # Show the updated plot, but without blocking
    plt.pause(0.01)

def callback(data, frame_count, time_info, status):
    wf.writeframes(data)
    plot_data(data)

    return (data, pyaudio.paContinue)

# start recording
stream = audio.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    # stream_callback=callback,
    frames_per_buffer=CHUNK
)

stream.start_stream()
while True:
    plot_data(stream.read(CHUNK, exception_on_overflow = False))
    time.sleep(0.1)


# while stream.is_active():
#     time.sleep(0.1)
# stream.close()

# audio.terminate()


# while(True):
#     print("recording")



    # stream.read(CHUNK)
    # frames = []
    # for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    #     data = stream.read(CHUNK)
    #     frames.append(data)
    # waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    # waveFile.setnchannels(CHANNELS)
    # waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    # waveFile.setframerate(RATE)
    # waveFile.writeframes(b''.join(frames))
    # waveFile.close()
    # spf = wave.open(WAVE_OUTPUT_FILENAME,'r')

    # #Extract Raw Audio from Wav File
    # signal = spf.readframes(-1)
    # signal = np.fromstring(signal, 'Int16')   
    # copy= signal.copy()