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
parser.add_argument("--peaks", "-p", action="store_true", help="Peak detection on spectrogram")
parser.add_argument("--save_file", "-sf", type=str,default=None, help="Name of csv filepath to upload data")
args = parser.parse_args()
def main():
	if(args.spectro):
		if(args.load_file != None):
			filepath = glob.glob('**/'+args.load_file,recursive=True) # Search for filename in current and all subdirectories
			if(len(filepath) == 0): raise ValueError(args.load_file+ ' not found!')
			spectrum,freqs = utils.spectrogram(filepath[0])
		else:
			raise Exception("--load_file argument missing!")
		if(args.save_file != None):
			utils.write_spectro(spectrum,freqs,SPECTRUM_CSV_DEST+args.save_file)
	if(args.peaks):
		if(args.load_file != None):
			filepath = glob.glob('**/'+args.load_file,recursive=True) # Search for filename in current and all subdirectories
			if(len(filepath) == 0): raise ValueError(args.load_file+ ' not found!')
			spectrum,freqs = utils.spectrogram(filepath[0],librosa_=True,plot=False)
			peaks,amps = utils.get_peaks(spectrum,freqs,plot=True)
			if(args.save_file != None):
				utils.write_peaks(peaks,amps,SPECTRUM_CSV_DEST+args.save_file)
		else:
			raise Exception("--load_file argument missing!")

if __name__ == '__main__':
	main()