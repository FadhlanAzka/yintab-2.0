# ================== GLOBAL SETTINGS ==================
import librosa

HOP_LENGTH   = 512
FRAME_LENGTH = 2048
FMIN         = librosa.note_to_hz('E2')   # ~82.41 Hz
FMAX         = librosa.note_to_hz('D6')   # ~1318.51 Hz
#FMAX         = librosa.note_to_hz('E6')   # ~1318.51 Hz

# Bandpass default (mengikuti range gitar)
BANDPASS_LOW  = FMIN
BANDPASS_HIGH = FMAX

# HTDemucs
USE_TWO_STEMS = True       # True: vocals/no_vocals; False: 4 stems
DEMUCS_MODEL  = "htdemucs" # default CLI model name

# Matplotlib figure layout
FIG_SIZE = (14, 8)
UPDATE_INTERVAL_MS = 50  # animator ~20 FPS
