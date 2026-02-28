import librosa
import numpy as np

def extract_mfcc(audio, sr, n_mfcc=40):
    """
    Extract Mel Frequency Cepstral Coefficients (MFCCs).
    """
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)

def extract_mel_spectrogram(audio, sr):
    """
    Extract Mel Spectrogram and convert to decibel scale.
    """
    mel = librosa.feature.melspectrogram(y=audio, sr=sr)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return np.mean(mel_db.T, axis=0)

def extract_chroma(audio, sr):
    """
    Extract Chroma feature representing the energy profile of pitch classes.
    """
    # use harmony component for chroma
    stft = np.abs(librosa.stft(audio))
    chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
    return np.mean(chroma.T, axis=0)

def extract_zcr(audio):
    """
    Extract Zero Crossing Rate (ZCR).
    """
    zcr = librosa.feature.zero_crossing_rate(y=audio)
    return np.mean(zcr.T, axis=0)

def extract_spectral_contrast(audio, sr):
    """
    Extract spectral contrast.
    """
    stft = np.abs(librosa.stft(audio))
    contrast = librosa.feature.spectral_contrast(S=stft, sr=sr)
    return np.mean(contrast.T, axis=0)

def extract_all_features(audio, sr):
    """
    Combine all extracted features into a single 1D feature vector.
    This is useful for feeding into traditional ML models or dense layers.
    For CNN architectures, a 2D Mel Spectrogram might be returned directly instead.
    """
    mfcc = extract_mfcc(audio, sr)
    mel = extract_mel_spectrogram(audio, sr)
    chroma = extract_chroma(audio, sr)
    zcr = extract_zcr(audio)
    contrast = extract_spectral_contrast(audio, sr)
    
    # Concatenate all features
    features = np.hstack([mfcc, mel, chroma, zcr, contrast])
    return features
