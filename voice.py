import numpy as np
import librosa
import scipy.signal
import os

# Load an audio file
def load_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    return y, sr

# Compute normalized cross-correlation
def compute_correlation(signal1, signal2):
    correlation = scipy.signal.correlate(signal1, signal2, mode='valid')
    correlation /= np.max(np.abs(correlation))  # Normalize
    return correlation

# Check if the correlation exceeds a threshold
def is_match(reference, test, threshold=0.7):
    correlation = compute_correlation(reference, test)
    max_corr = np.max(correlation)
    print(f"Maximum correlation: {max_corr:.2f}")
    return max_corr > threshold

# Main function
def main():
    reference_file = 'reference.wav'
    test_file = 'test.wav'

    if os.path.exists(reference_file) and os.path.exists(test_file):
        ref_signal, ref_sr = load_audio(reference_file)
        test_signal, test_sr = load_audio(test_file)

        # Resample if needed
        if ref_sr != test_sr:
            test_signal = librosa.resample(test_signal, orig_sr=test_sr, target_sr=ref_sr)

        if is_match(ref_signal, test_signal):
            print("✅ Voice command matched! Executing command...")
            # Place your command logic here
        else:
            print("❌ Voice command did not match.")
    else:
        print("Please ensure 'reference.wav' and 'test.wav' exist in the current directory.")

if __name__ == "__main__":
    main()
