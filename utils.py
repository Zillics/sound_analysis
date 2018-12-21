import glob
import matplotlib.pyplot as plot
import seaborn as sns
import pandas as pd
from scipy.io import wavfile
import numpy as np

def spectrogram(filename):
	sns.set()
	sns.set_style("whitegrid", {'axes.grid' : False})
	# Search for filename in current and all subdirectories
	filepath = glob.glob('**/'+filename,recursive=True)
	if(len(filepath) == 0): raise ValueError(filename + ' not found!')

	# Read the wav file (mono)
	samplingFrequency, signalData = wavfile.read(filepath[0])

	# Plot the signal read from wav file
	plot.subplot(211)
	plot.title('Spectrogram of ' + filename)
	plot.plot(signalData)
	plot.xlabel('Sample')
	plot.ylabel('Amplitude')
	plot.subplot(212)
	spectrum,freqs,_,_ = plot.specgram(signalData,Fs=samplingFrequency)#,scale='dB')
	plot.xlabel('Time')
	plot.ylabel('Frequency')
	plot.show()
	return spectrum,freqs

def write_spectro(spectrum,freqs,filepath,norm=True):
	mean_amps = spectrum.mean(axis=1)
	
	if(norm):
		simple_spectro = np.array([freqs,normalize(mean_amps)]).transpose()
	else:
		simple_spectro = np.array([freqs,mean_amps]).transpose()
	
	df = pd.DataFrame(data=simple_spectro,index=None)
	df.to_csv(filepath, sep=',', index=False,header=False)

def normalize(values,norm_0=True):
	_min = values.min()
	_max = values.max()
	if(norm_0):
		norm_values = values/_max
	else:
		norm_values = (_values - _min)/(_max - _min)
	return norm_values
