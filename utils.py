import glob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.io import wavfile
from scipy.signal import find_peaks,peak_prominences
import numpy as np
import librosa
import librosa.display
import librosa.onset
import time

sns.set()
sns.set_style("whitegrid", {'axes.grid' : False})

MIN_PROMINENCE = 1

# Find file(s) recursively through all subdirectories below current. 
# IF all_files is enabled, returns list of all files with extension
# ELSE Returns first file that is found with name filename
def find_file(filename,all_files=False,extension='.wav'):
	if(all_files):
		filepath = glob.glob(filename+'/*'+extension)
		if(len(filepath) == 0): raise ValueError('No .wav files found from '+filename)
		return filepath
	else:
		filepath = glob.glob('**/'+filename,recursive=True) # Search for filename in current and all subdirectories
		if(len(filepath) == 0): raise ValueError(filename + ' not found!')
		return filepath[0]

def visualize(filepath,raw=False,harm_perc=False,spectrogram=False,harm_spectr=True,perc_spectr=True):
	y, sr_ = librosa.load(filepath)
	features = [raw,harm_perc,spectrogram,harm_spectr,perc_spectr]
	subplot_n = np.array(features).sum()
	fig = plt.figure(figsize=(20,15))
	ax1 = plt.subplot(subplot_n, 1, 1)
	i = 1
	if(raw):
		plt.title('Raw audio wave')
		librosa.display.waveplot(y, sr=sr_)
		i += 1
	if(harm_perc):
		plt.subplot(subplot_n, 1, i)
		plt.title('Raw audio wave separated into harmonic and percussive components')
		y_harm, y_perc = librosa.effects.hpss(y)
		librosa.display.waveplot(y_harm, sr=sr_, alpha=0.25)
		librosa.display.waveplot(y_perc, sr=sr_, color='r', alpha=0.5)
		i += 1
	if(spectrogram):
		plt.subplot(subplot_n, 1, i)
		plt.title('Spectrogram of log scale')
		D = librosa.stft(y)
		magnitude, phase = librosa.magphase(D)
		rp = np.max(np.abs(D))
		librosa.display.specshow(librosa.amplitude_to_db(D, ref=rp), y_axis='log')
		i += 1
	if(harm_spectr):
		plt.subplot(subplot_n, 1, i)
		plt.title('Spectrogram of harmonic_components (log scale)')
		D = librosa.stft(y)
		D_harmonic, D_percussive = librosa.decompose.hpss(D)
		rp = np.max(np.abs(D_harmonic))
		librosa.display.specshow(librosa.amplitude_to_db(D_harmonic, ref=rp), y_axis='log')
		i += 1
	if(perc_spectr):
		plt.subplot(subplot_n, 1, i)
		plt.title('Spectrogram of percussive_components (log scale)')		
		rp = np.max(np.abs(D_percussive))
		librosa.display.specshow(librosa.amplitude_to_db(D_percussive, ref=rp), y_axis='log')
		i += 1

	plt.suptitle('Visualization for '+filepath)
	plt.show()

def visualize_S(S,freqs):
	plt.figure()
	librosa.display.specshow(S, y_axis='log')
	plt.show()

def fit_freqs(S,freqs,plot=False):
	freqs,amps,freq_idx = get_harmonics(S,freqs,plot=False)
	print(freqs.shape)
	print(amps.shape)
	print(amps)
	if(plot):
		plt.ion()
	for i in range(freq_idx.shape[0]):
		if(plot):
			fig, ax_list_ = plt.subplots(3,1)
		else:
			ax_list_ = []
		S_freq = S[freq_idx[i],:]
		try:
			fit_ARIMA(S_freq,ax_list=ax_list_)
		except Exception as e:
			print(i," Failed!")
			print(e)
		if(plot):
			fig.set_size_inches(20,10)
			plt.legend()
			plt.suptitle('Freq index: '+str(freq_idx[i])+'\nFrequency: '+str(int(freqs[i]))+'Hz')
			plt.draw()
			plt.pause(0.001)
			# Wait 1s and clear axis
			time.sleep(3)
			plt.close('all')

def fit_ARIMA(x,p=5,d=2,q=0,ax_list=[]):
	t_series = pd.Series(data=x)
	model = ARIMA(t_series,order=(p,d,q))
	model_fit = model.fit(disp=0)
	residuals = pd.DataFrame(model_fit.resid)
	if(len(ax_list) == 3):
		ax_list[0].plot(range(x.shape[0]),x)
		ax_list[0].set_title('Amplitude of frequency over time')
		residuals.plot(ax=ax_list[1])
		ax_list[1].set_title('Residual of ARIMA model over time')
		residuals.plot(kind='kde',ax=ax_list[2])
		ax_list[2].set_title('Distribution of residuals for ARIMA model')

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
	return freqs[peaks_idx],means[peaks_idx],peaks_idx

# Input: Spectrogram S, frequencies freqs of that Sprectrogram, sample rate sr_ of Spectrogram
# Returns ranges of Attack, Sustain and Release for a soundwave of one(!) note pressed
# Customized ADSR range estimator for one(!) note soundwaves. Attack is here defined as A+D in ADSR
# Could easily be extended to output A and D separately as well in the future.(Something on TODO list) 
# How algorithm works (and why) might get clearer by looking at the plots.
# TODO:
# 1. Output ranges of A and D separately
# 2. Output fourth range between current release start and when curve starts to stabilize
def get_adsr(S,freqs,sr_,filename='File?',plot=True):
	D_harmonic, D_percussive = librosa.decompose.hpss(S)
	means = np.mean(D_harmonic,axis=0)
	grad = np.gradient(means)
	# Total variance and rolling variance
	grad_var = np.var(grad)
	df = pd.DataFrame(data=grad)
	rol_var = df.rolling(window=10).var()[0]
	# Find first peak of gradient
	p1 = np.argmax(grad)
	# Find where the curve changes direction after p1
	p2_ = p1 + np.argmax(grad[p1:] <= 0)
	# Find where the rolling variance decreases to total variance
	p2 = p2_ + np.argmax(rol_var[p2_:].values <= grad_var)
	# Find lowest peak of gradient
	p3 = p2 + np.argmin(grad[p2:])
	# If p2 is lowest point after attack -> release starts after attack
	if(p2==p3):
		attack = (0,p2)
		sustain = (p2,p2) # No sustain
		release = (p2,grad.shape[0])
	else:
		# If p2 is not lowest point after attack -> Sustain starts after attack.
		# Find where the curve changes direction before p3
		p4 = p3 - np.argmax(np.flip(grad[p2:p3]) >= grad[p2:p3].mean())
		attack = (0,p2)
		sustain = (p2,p4)
		release = (p4,grad.shape[0])
	
	if(plot):
		rp = np.max(np.abs(D_harmonic))
		fig = plt.figure(figsize=(20,15))
		ax1 = plt.subplot(3, 1, 1)
		plt.title("Spectrogram (log scale)")
		librosa.display.specshow(librosa.amplitude_to_db(D_harmonic, ref=rp), y_axis='log')
		plt.subplot(3, 1, 2, sharex=ax1)
		plt.title('Gradient of mean amplitudes of all harmonic frequencies over time')
		plt.plot(grad)
		plt.scatter(p1,grad[p1],color='r',label='p1')
		plt.scatter(p2,grad[p2],color='g',label='p2')
		plt.scatter(p3,grad[p3],color='black',label='p3')
		if( not (p2==p3)):
			plt.scatter(p4,grad[p4],color='b',label='p4')
			plt.vlines(attack[1],grad.min(),grad.max(),label='attack end/sustain start',color='b',linestyle='--')
			plt.vlines(sustain[1],grad.min(),grad.max(),label='sustain end/release start',color='r',linestyle='--')
		else:
			plt.vlines(attack[1],grad.min(),grad.max(),label='attack end/release start',color='b',linestyle='--')
		plt.xlabel('Time [sample index]')
		plt.ylabel('Gradient of mean of amplitudes of all harmonic frequencies')
		plt.legend()	
		plt.subplot(3,1,3,sharex=ax1)
		plt.title('Variances')
		plt.plot(rol_var,label='Rolling variance of gradient')
		plt.hlines(grad_var, 0, grad.shape[0], color='r',label='Total variance of gradient')
		plt.xlabel('Time [sample index]')
		plt.ylabel('Variance of upper gradient')
		plt.legend()
		plt.axis('tight')
		fig.suptitle('ADSR envelope estimated for '+filename)
		plt.show()
	return attack,sustain,release
