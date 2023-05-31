import os
import random
import math
import json

content = "/media/goku/4b66c306-b38b-4701-9bd5-fd5c65a905fd/asjad.s/EGO4D"
with open(content+'/vq2d_cvpr/vq2d/config.yaml', 'r') as file :
  filedata = file.read()
filedata = filedata.replace('/private/home/sramakri/Research/Ego4D/code/episodic-memory-internal/', content+'/')
with open(content+'/vq2d_cvpr/vq2d/config.yaml', 'w') as file:
  file.write(filedata)
