import glob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.io import wavfile
from scipy.signal import find_peaks,peak_prominences
import numpy as np
import librosa
import librosa.display
import librosa.onset

sns.set()
sns.set_style("whitegrid", {'axes.grid' : False})

MIN_PROMINENCE = 1 # Minimum prominence required for a frequency to be considered a harmonic. Used in peak detection

def spectrogram(filepath,librosa_=True,mel=False,plot=True):
	if(not librosa_):
		# Read the wav file (mono)
		samplingFrequency, signalData = wavfile.read(filepath)
		spectrum,freqs,_,_ = plot.specgram(signalData,Fs=samplingFrequency)#,scale='dB')
		# Plot the signal read from wav file
		if(plot):
			plt.subplot(211)
			plt.title('Spectrogram of ' + filepath)
			plt.plot(signalData)
			plt.xlabel('Sample')
			plt.ylabel('Amplitude')
			plt.subplot(212)
			plt.xlabel('Time')
			plt.ylabel('Frequency')
			plt.show()
	else:
		if(not mel):
			y,sr_ = librosa.load(filepath)
			n_fft_ = 2050 # FFT window size
			D = librosa.stft(y,n_fft=n_fft_)
			magnitude, phase = librosa.magphase(D)
			freqs = librosa.core.fft_frequencies(sr=sr_,n_fft=n_fft_)
			spectrum = magnitude
		else:
			y, sr_ = librosa.load(filepath,sr=None)
			print("Creating spectrogram with sampling rate", sr, "Hz....")
			fmax_ = sr_/2
			S = librosa.feature.melspectrogram(y=y, sr=sr_,fmax=fmax_)
			spectrum = S
			freqs = librosa.core.mel_frequencies(n_mels=S.shape[0],fmax=fmax_)
			if(plot):
				plt.figure(figsize=(20, 8))
				librosa.display.specshow(librosa.power_to_db(S,ref=np.max),y_axis='mel',fmax=fmax_,x_axis='time')
				plt.colorbar(format='%+2.0f dB')
				plt.title('Mel spectrogram')
				plt.tight_layout()
				plt.show()
	return spectrum,freqs,sr_

def write_spectro(spectrum,freqs,filepath,norm=True):
	mean_amps = spectrum.mean(axis=1)
	if(norm):
		simple_spectro = np.array([freqs,normalize(mean_amps)]).transpose()
	else:
		simple_spectro = np.array([freqs,mean_amps]).transpose()
	
	df = pd.DataFrame(data=simple_spectro,index=None)
	print("Writing to file ",filepath,".....")
	df.to_csv(filepath, sep=',', index=False,header=False)
	print("File written successfully!")

def write_peaks(peak_freqs, peak_amps,filepath,norm=True):
	if(norm):
		freq_amp = np.array([peak_freqs,normalize(peak_amps)]).transpose()
	else:
		freq_amp = np.array([peak_freqs,peak_amps]).transpose()
	df = pd.DataFrame(data=freq_amp,index=None)
	print("Writing to file ", filepath,"....")
	df.to_csv(filepath,sep=',',index=False,header=False)
	print("File written successfully!")

def normalize(values,norm_0=True):
	_min = values.min()
	_max = values.max()
	if(norm_0):
		norm_values = values/_max
	else:
		norm_values = (_values - _min)/(_max - _min)
	return norm_values

# Spectrogram S (shape=(n_freqs,n_time)), frequencies freqs (shape=(n_freqs))
# Returns freqs,amps
# freqs = list of frequencies of each peak (shape=(n_peaks))
# amps = mean amplitudes of corresponding peak frequencies (shape=(n_peaks))
def get_peaks(S,freqs,plot=False,harm_sep=True):
	S_db = librosa.power_to_db(S,ref=np.max)
	means_db = S_db.mean(axis=1)
	means_0 = means_db - means_db.min()
	peaks_idx, _ = find_peaks(means_0,prominence=MIN_PROMINENCE, height=0)
	if(plot):	
		plt.figure(figsize=(20,8))
		plt.plot(freqs,means_db)
		plt.scatter(freqs[peaks_idx],means_db[peaks_idx])
		plt.show()
	means = S.mean(axis=1)
	return freqs[peaks_idx],means[peaks_idx]
# S: spectrogram (shape=(magnitude of frequencies,time))
def get_harmonics(S,freqs,plot=True):
	D_harmonic, D_percussive = librosa.decompose.hpss(S)
	print(D_harmonic)
	rp = np.max(np.abs(S))
	if(plot):
		plt.figure(figsize=(12, 8))

		plt.subplot(3, 1, 1) #Pre-compute a global reference power from the input spectrum
		librosa.display.specshow(librosa.amplitude_to_db(S, ref=rp), y_axis='log')
		plt.colorbar()
		plt.title('Full spectrogram')

		plt.subplot(3, 1, 2)
		librosa.display.specshow(librosa.amplitude_to_db(D_harmonic, ref=rp), y_axis='log')
		plt.colorbar()
		plt.title('Harmonic spectrogram')

		plt.subplot(3, 1, 3)
		librosa.display.specshow(librosa.amplitude_to_db(D_percussive, ref=rp), y_axis='log', x_axis='time')
		plt.colorbar()
		plt.title('Percussive spectrogram')
		plt.tight_layout()

		plt.show()
	means_db = librosa.amplitude_to_db(D_harmonic,ref=rp).mean(axis=1)
	means_0 = means_db - means_db.min()
	peaks_idx, _ = find_peaks(means_0,prominence=MIN_PROMINENCE, height=0)
	if(plot):	
		plt.figure(figsize=(20,8))
		plt.plot(freqs,means_db)
		plt.scatter(freqs[peaks_idx],means_db[peaks_idx])
		plt.show()
	means = D_harmonic.mean(axis=1)
	return freqs[peaks_idx],means[peaks_idx]

def get_adsr(S,freqs,sr_,plot=True):
	D_harmonic, D_percussive = librosa.decompose.hpss(S)
	print(D_harmonic)
	print(D_harmonic.shape)
	means = np.mean(D_harmonic,axis=0)
	grad = np.gradient(means)
	# Find first peak of gradient
	p1 = np.argmax(grad)
	# Find lowest peak of gradient
	p2 = np.argmin(grad)
	# Find where the curve changes direction after p1
	p3 = p1 + np.argmax(grad[p1:p2] <= 0)
	# Find where the curve changes direction after p2
	p4 = p2 - np.argmax(np.flip(grad[p1:p2]) >= 0)
	# Determine whether it has Sustain or Release

	grad_2 = np.gradient(grad)
	grad_3 = np.gradient(grad_2)
	grad_4 = np.gradient(grad_3)
	peak_idx,_ = find_peaks(grad_3)

	df = pd.DataFrame(data=grad)
	rol_mean = df.rolling(window=10).mean()
	rol_mean_g = rol_mean.diff()
	rol_var = df.rolling(window=10).mean()
	if(plot):
		rp = np.max(np.abs(D_harmonic))
		plt.figure(figsize=(20,15))
		ax1 = plt.subplot(3, 1, 1)
		plt.title("Spectrogram (log scale) and Mean of gradients for each harmonic frequency")
		librosa.display.specshow(librosa.amplitude_to_db(D_harmonic, ref=rp), y_axis='log')
		plt.subplot(3, 1, 2, sharex=ax1)
		plt.plot(grad)
		#plt.subplot(3,1,3,sharex=ax1)
		#plt.plot(oenv)
		#plt.vlines(onset_bt, 0, oenv.max(), label='Backtracked', color='r')
		plt.scatter(p1,grad[p1],color='r')
		plt.scatter(p2,grad[p2],color='b')
		plt.scatter(p3,grad[p3],color='g')
		plt.scatter(p4,grad[p4],color='y')		
		plt.subplot(3,1,3,sharex=ax1)
		plt.plot(rol_var)
		#plt.scatter(max_peaks,means[max_peaks],color='b')
		#plt.scatter(min_peaks,means[min_peaks],color='r')
		plt.axis('tight')
		plt.show()
