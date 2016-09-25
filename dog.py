from scipy.fftpack import fft2
from scipy.fftpack import ifft2
import cv2
import numpy
import matplotlib.pyplot as plt
import sys
import os
import math



pitch = 0.3
depth = 30
path = sys.argv[1]
img = cv2.imread(path, 0)
w = img.shape[1]
h = img.shape[0]
size = max(w,h)
img = cv2.resize(img,(size,size))
output = sys.argv[2]



if not os.path.exists(output):
	os.mkdir(output)

if not os.path.exists(output+"/blar"):
	os.mkdir(output+"/blar")

if not os.path.exists(output+"/dog"):
	os.mkdir(output+"/dog")



for i in range(depth):

	stddev = (i+1)*pitch
	print stddev

	gf = numpy.zeros((size,size))

	for y in range(size):
		for x in range(size):
			u = x-size/2
			v = y-size/2
			gf[y,x] = math.exp(-(u*u+v*v)/(2*pow(stddev,2)))

	gf /= numpy.sum(gf)
	F = fft2(gf)

	if i == 0:
		F_gf = F
	else:
		F_gf = numpy.r_[F_gf,F]


F_img = fft2(img)
F_gf = F_gf.reshape((depth,size,size))



for i in range(depth):
	F = F_img.copy()
	for y in range(size):
		for x in range(size):
			F[y,x] *= F_gf[i,y,x]
			
	f = numpy.real(ifft2(F))
	blar = f.copy()
	blar[0:size/2,0:size/2] = f[size/2:size,size/2:size]
	blar[size/2:size,size/2:size] = f[0:size/2,0:size/2]
	blar[0:size/2,size/2:size] = f[size/2:size,0:size/2]
	blar[size/2:size,0:size/2] = f[0:size/2,size/2:size]

	cv2.imwrite(output+"/blar/"+str(i)+".jpg", blar)

	print i

	if i == 0:
		space = blar
	else:
		space = numpy.r_[space,blar]

space = space.reshape((depth,size,size))


for i in range(depth-1):
	dog = numpy.abs(space[i]-space[i+1].astype(numpy.float64))

	if i == 0:
		dogs = dog
	else:
		dogs = numpy.r_[dogs,dog]

	mi = numpy.min(dog)
	dog -= mi
	ma = numpy.max(dog)
	dog *= (255.0/ma)
	dog = dog.astype(numpy.uint8)
	cv2.imwrite(output+"/dog/"+str(i)+".jpg", dog)


dogs = dogs.reshape((depth-1,size,size))


surf = numpy.zeros((size,size))
for y in range(size):
	for x in range(size):
		surf[y,x] = (numpy.argmax([dogs[j+1,y,x] for j in range(depth-1-1)])+1)*pitch
		


plt.imshow(surf)
plt.colorbar()
plt.savefig(output+"/scale.jpg")
plt.show()