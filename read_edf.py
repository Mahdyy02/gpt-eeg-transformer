import mne

# Path to your EDF file
edf_file = fr"tuh_eeg\train\aaaaaamp\s003_2003\02_tcp_le\aaaaaamp_s003_t000.edf"

# Load the EDF file
raw = mne.io.read_raw_edf(edf_file, preload=True)

# Print basic info about the file
print("=== EDF File Info ===")
print(raw.info)

# List all channel names
print("\n=== Channels ===")
print(raw.ch_names)

# Print number of samples, sampling frequency, duration
print("\n=== Data Summary ===")
n_samples = raw.n_times
sfreq = raw.info['sfreq']
duration = n_samples / sfreq
print(f"Number of samples: {n_samples}")
print(f"Sampling frequency: {sfreq} Hz")
print(f"Duration: {duration:.2f} seconds")

# Access the data as a NumPy array
data = raw.get_data()  # shape: (n_channels, n_samples)
print("\n=== Data shape ===")
print(data.shape)

# Print first 5 seconds of each channel
print("\n=== First 5 seconds of data ===")
samples_5s = int(5 * sfreq)
print(data[:, :samples_5s])

# # Optionally, plot the signals
raw.plot(scalings='auto', show=True)
