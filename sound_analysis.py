#import the pyplot and wavfile modules 
import argparse
import utils
import glob
import librosa
import numpy as np
import os

SPECTRUM_CSV_DEST = "../../C++/stk_adaptive_synth/src/synth_files/"
DEST_DIR = "../../C++/stk_adaptive_synth/src/synth_files/"

parser = argparse.ArgumentParser()
parser.add_argument("--spectro", "-sp", action="store_true", help="Spectrogram")
parser.add_argument("--visualize", "-vi",action="store_true", help="Visualize wavfile")
parser.add_argument("--load_file", "-lf", type=str,default=None, help="Name of wav file to analyze")
parser.add_argument("--load_dir", "-ld", type=str,default=None, help="Name of directory of wav file(s)")
parser.add_argument("--peaks", "-p", action="store_true", help="Peak detection on spectrogram (Outdated)")
parser.add_argument("--harm", "-hr", action="store_true", help="Get harmonics from sound file")
parser.add_argument("--analyze", "-an", action="store_true",help="Deep analysis and feature extraction for soundwave")
parser.add_argument("--adsr",action="store_true", help="Estimate time ranges for Attack,Sustain and Release")
parser.add_argument("--resynthesize_static",action="store_true",help="Resynthesize with static amplitudes for each frequency")
parser.add_argument("--resynthesize_dynamics",action="store_true",help="Resynthesize with dynamic amplitudes for each frequency")
parser.add_argument("--plot", "-plt", action="store_true", help="Plot everything")
parser.add_argument("--save_file", "-sf", type=str,default=None, help="Name of csv filepath to upload data")
parser.add_argument("--_fit_noise",action="store_true")
args = parser.parse_args()
def main():
	if(args.resynthesize_static):
		if(args.load_file != None):
			filepath = utils.find_file(args.load_file)
			spectrum,freqs,_ = utils.spectrogram(filepath,plot=args.plot)
			freqs,amps,freq_idx = utils.get_harmonics(spectrum,freqs,plot=args.plot)
			noise_mean,noise_std = utils.fit_noise(spectrum,freqs,plot=args.plot)
			if(args.save_file != None):
				save_path = DEST_DIR+args.save_file
			else:
				save_path = DEST_DIR+args.load_file[:-4] + '_resynthesized.csv'
			noise_path = save_path[:-4]+'_noise.csv'
			utils.write_peaks(freqs,amps,save_path,norm=True)
			utils.write_noise(noise_mean,noise_std,noise_path)
		else:
			if(args.load_dir != None):
				filepath = utils.find_file(args.load_dir,all_files=True,extension='.wav')
				for file in filepath:
					try:
						spectrum,freqs,_ = utils.spectrogram(file,plot=args.plot)
						freqs,amps,freq_idx = utils.get_harmonics(spectrum,freqs,plot=args.plot)
						save_path = DEST_DIR + os.path.basename(file) + '_resynthesized.csv'
						utils.write_peaks(freqs,amps,save_path,norm=True)
					except KeyboardInterrupt:
						raise KeyboardInterrupt("Ctrl-c pressed!")
					except Exception as e:
						print(e)
			else:
				raise Exception("--load_file or --load_dir argument missing!")
	if(args._fit_noise):
		if(args.load_file != None):
			filepath = utils.find_file(args.load_file)
			S,freqs,sr = utils.spectrogram(filepath,librosa_=True,mel=False,plot=False)
			attack,sustain,release = utils.get_adsr(S,freqs,sr,filename=filepath,plot=False)
			S_sustain=S[:,sustain[0]:sustain[1]]
			#utils.visualize_S(S_sustain,sr)
			noise_mean, noise_std = utils.fit_noise(S_sustain,freqs,args.plot)
		else:
			raise Exception("--load_file argument missing!")
	if(args.spectro):
		if(args.load_file != None):
			filepath = glob.glob('**/'+args.load_file,recursive=True) # Search for filename in current and all subdirectories
			if(len(filepath) == 0): raise ValueError(args.load_file+ ' not found!')
			spectrum,freqs = utils.spectrogram(filepath[0],plot=args.plot)
		else:
			raise Exception("--load_file argument missing!")
		if(args.save_file != None):
			utils.write_spectro(spectrum,freqs,SPECTRUM_CSV_DEST+args.save_file)
	if(args.peaks):
		if(args.load_file != None):
			filepath = glob.glob('**/'+args.load_file,recursive=True) # Search for filename in current and all subdirectories
			if(len(filepath) == 0): raise ValueError(args.load_file+ ' not found!')
			spectrum,freqs = utils.spectrogram(filepath[0],librosa_=True,plot=args.plot)
			peaks,amps = utils.get_peaks(spectrum,freqs,plot=args.plot)
			if(args.save_file != None):
				utils.write_peaks(peaks,amps,SPECTRUM_CSV_DEST+args.save_file)
		else:
			raise Exception("--load_file argument missing!")
	if(args.harm):
		if(args.load_file != None):
			filepath = glob.glob('**/'+args.load_file,recursive=True) # Search for filename in current and all subdirectories
			if(len(filepath) == 0): raise ValueError(args.load_file+ ' not found!')
			S,freqs,sr = utils.spectrogram(filepath[0],librosa_=True,mel=False,plot=args.plot)
			peaks,amps,_ = utils.get_harmonics(S,freqs)
			if(args.save_file != None):
				utils.write_peaks(peaks,amps,SPECTRUM_CSV_DEST+args.save_file)
		else:
			raise Exception("--load_file argument missing!")
	if(args.analyze):
		if(args.load_file != None):
			filepath = utils.find_file(args.load_file)
			S,freqs,sr = utils.spectrogram(filepath, librosa_=True,mel=False,plot=args.plot)
			attack,sustain,release = utils.get_adsr(S,freqs,sr,filename=filepath,plot=False)
			# Take a look at sustain part
			S_sustain = S[:,sustain[0]:sustain[1]]
			utils.fit_freqs(S_sustain,freqs,plot=args.plot)
	if(args.adsr):
		if(args.load_file != None):
			filepath = utils.find_file(args.load_file)
			S,freqs,sr = utils.spectrogram(filepath,librosa_=True,mel=False,plot=args.plot)
			attack,sustain,release = utils.get_adsr(S,freqs,sr,filename=filepath,plot=args.plot)
			print(attack)
			print(sustain)
			print(release)
		else:
			if(args.load_dir != None):
				filepath = utils.find_file(args.load_dir,all_files=True,extension='.wav')
				for file in filepath:
					try:
						S,freqs,sr = utils.spectrogram(file,librosa_=True,mel=False,plot=args.plot)
						attack,sustain,release=utils.get_adsr(S,freqs,sr,filename=file,plot=args.plot)
						print('Attack: ', attack)
						print('Sustain: ',sustain)
						print('Release: ',release)
					except KeyboardInterrupt:
						raise KeyboardInterrupt("Ctrl+c pressed!")
					except Exception as e:
						print(e)
			else:
				raise Exception("--load_file or --load_dir argument missing!")
	if(args.visualize):
		if(args.load_file != None):
			filepath = utils.find_file(args.load_file)
			utils.visualize(filepath)

if __name__ == '__main__':
	main()