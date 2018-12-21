#import the pyplot and wavfile modules 
import argparse
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--spectro", "-sp", action="store_true", help="Spectrogram")
parser.add_argument("--load_file", "-lf", type=str,default=None, help="Name of wav file to analyze")
parser.add_argument("--save_file", "-sf", type=str,default=None, help="Name of csv filepath to upload data")
args = parser.parse_args()

def main():
	if(args.spectro):
		if(args.load_file != None):
			spectrum,freqs = utils.spectrogram(args.load_file)
		else:
			raise Exception("--load_file argument missing!")
		if(args.save_file != None):
			utils.write_spectro(spectrum,freqs,args.save_file)

if __name__ == '__main__':
	main()