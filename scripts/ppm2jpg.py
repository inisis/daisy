import argparse
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--InFile', required=True, help='Input File Name')
parser.add_argument('--OutFile', required=True, help='Output File Name')

args = parser.parse_args()

im = Image.open(args.InFile)
im.save(args.OutFile)
