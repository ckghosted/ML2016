import sys
from PIL import Image

fileName = sys.argv[1]
im = Image.open(fileName)
im2 = im.transpose(Image.ROTATE_180)
im2.save('ans2.png')
