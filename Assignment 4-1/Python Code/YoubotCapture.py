# -*- coding: utf-8 -*-
"""
Created on Mon May 11 15:30:44 2015

@author: Elladyr
"""

import numpy as np
import matplotlib.pyplot as plt

#cams = AP.list_cameras() #lifecam19, lifecam20 and lifecam22
#img = AP.get_camera_image(cams[1])
#plt.imshow(image[0][:,:,0])

idx = 1

fpath = 'E:/Studies/Semester 4/Scientific Experimentation and Evaluation/Lecture 5/Youbot experiment/forward_exp'
fname = '/marker_22203_exp_'

array = None
full = []

#print output

#for idx in xrange(1, 3):
#    print fpath+fname+str(idx)+'.txt'

for idx in xrange(1,22):

    a = str(fpath+fname+str(idx)+'.txt')
    
    with open(a) as f:
        content = f.read().splitlines()
        f.close()
    
    for i in xrange(0, len(content)):
        temp = content[i].split()
        if array is None:
            array = np.asarray(temp[3:])
        else:
            array = np.vstack((array, np.asarray(temp[3:])))
    
    full.append(array)
    array = None

for i in xrange(0, len(full)):
    plt.plot(full[i][:,1], full[i][:,0])

plt.title('Starting pose')
plt.xlabel('Y coordinates')
plt.ylabel('X coordinates')
