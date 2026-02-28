import librosa
import numpy as np

def load_audio(file_path, sr=22050):
    """
    Load an audio file and resample it if necessary.
    """
    audio, sample_rate = librosa.load(file_path, sr=sr)
    return audio, sample_rate

def trim_silence(audio, top_db=30):
    """
    Trim leading and trailing silence from an audio signal.
    """
    trimmed_audio, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed_audio

def add_white_noise(audio, noise_factor=0.005):
    """
    Add random white noise to the audio signal for data augmentation.
    """
    noise = np.random.randn(len(audio))
    augmented_audio = audio + noise_factor * noise
    return augmented_audio

def pitch_shift(audio, sr, n_steps=2):
    """
    Shift the pitch of an audio signal by n_steps.
    """
    return librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=n_steps)

def time_stretch(audio, rate=1.2):
    """
    Time-stretch an audio signal by a given rate without altering pitch.
    """
    return librosa.effects.time_stretch(y=audio, rate=rate)

def pad_or_truncate(audio, max_length=100000):
    """
    Pad with zeros or truncate the audio signal to ensure consistent length.
    """
    if len(audio) > max_length:
        return audio[:max_length]
    else:
        padding = max_length - len(audio)
        return np.pad(audio, (0, padding), 'constant')

def preprocess_pipeline(file_path, sr=22050, max_length=100000, augment=False):
    """
    End-to-end preprocessing pipeline combining loading, trimming, 
    optional augmentation, and length standardization.
    """
    audio, sr = load_audio(file_path, sr=sr)
    audio = trim_silence(audio)
    
    if augment:
        # Randomly choose an augmentation technique
        aug_type = np.random.choice(['noise', 'pitch', 'stretch', 'none'], p=[0.3, 0.3, 0.2, 0.2])
        if aug_type == 'noise':
            audio = add_white_noise(audio)
        elif aug_type == 'pitch':
            audio = pitch_shift(audio, sr)
        elif aug_type == 'stretch':
            audio = time_stretch(audio)
            
    audio = pad_or_truncate(audio, max_length=max_length)
    return audio, sr
