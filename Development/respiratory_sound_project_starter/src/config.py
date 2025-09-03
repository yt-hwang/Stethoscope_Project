# Central configuration for audio and features

SR = 16000                 # sample rate (Hz)
FRAME_MS = 25              # STFT window size (ms)
HOP_MS = 10                # STFT hop size (ms)
N_FFT = 512                # >= frame_len; 25ms @ 16kHz = 400 samples → 512 is safe
N_MELS = 64
FMIN = 50
FMAX = 4000                # upper bound for wheeze energy focus

# Derived
FRAME_LEN = int(SR * FRAME_MS / 1000)
HOP_LEN = int(SR * HOP_MS / 1000)

# Training
NUM_CLASSES = 3            # breathing, wheezing, noise
SEED = 1337
