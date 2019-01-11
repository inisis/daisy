import argparse
from PIL import Image
import os, sys

#parser = argparse.ArgumentParser()
#parser.add_argument('--InFile', required=True, help='Input File Name')
#parser.add_argument('--OutFile', required=True, help='Output File Name')

#args = parser.parse_args()

img_path = sys.argv[1]
h = sys.argv[2]
w = sys.argv[3]

img_name = os.path.basename(img_path)
img = Image.open(img_path)
img = img.resize((int(w), int(h)))
img.save(img_name[:-4]+'.ppm')
