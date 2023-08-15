from PIL import Image as ImagePIL
import os
infile = "cat\\5.jpg"
outfile = "cat\\55.jpg"

img = ImagePIL.open(infile)
img2 = img.crop((70,0,278,208))
img2=img2.convert('RGB')
img2.save(outfile, dpi=(100, 100)) #想要设定的dpi值