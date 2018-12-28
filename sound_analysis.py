#import the pyplot and wavfile modules 
import argparse
import utils
import glob
import librosa
import numpy as np

SPECTRUM_CSV_DEST = "../../C++/stk_adaptive_synth/src/synth_files/"

parser = argparse.ArgumentParser()
parser.add_argument("--spectro", "-sp", action="store_true", help="Spectrogram")
parser.add_argument("--load_file", "-lf", type=str,default=None, help="Name of wav file to analyze")
parser.add_argument("--peaks", "-p", action="store_true", help="Peak detection on spectrogram (Outdated?)")
parser.add_argument("--harm", "-hr", action="store_true", help="Get harmonics from sound file")
parser.add_argument("--adsr",action="store_true", help="Estimate time ranges for Attack,Sustain and Release")
parser.add_argument("--plot", "-plt", action="store_true", help="Plot everything")
parser.add_argument("--save_file", "-sf", type=str,default=None, help="Name of csv filepath to upload data")
args = parser.parse_args()
def main():
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
			peaks,amps = utils.get_harmonics(S,freqs)
			if(args.save_file != None):
				utils.write_peaks(peaks,amps,SPECTRUM_CSV_DEST+args.save_file)
		else:
			raise Exception("--load_file argument missing!")
	if(args.adsr):
		if(args.load_file != None):
			filepath = glob.glob('**/'+args.load_file,recursive=True) # Search for filename in current and all subdirectories
			if(len(filepath) == 0): raise ValueError(args.load_file+ ' not found!')
			S,freqs,sr = utils.spectrogram(filepath[0],librosa_=True,mel=False,plot=args.plot)
			utils.get_adsr(S,freqs,sr,plot=args.plot)

if __name__ == '__main__':
	main()