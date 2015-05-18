# -*- coding: utf-8 -*-
"""
Created on Mon May 11 15:30:44 2015

@author: Elladyr
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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

#color=iter(cm.rainbow(np.linspace(0,1,21)))
colors = plt.cm.Set1(np.linspace(0, 1, 21))

processed = np.zeros((30,6,21))

for i in xrange(0, len(full)):
    plt.plot(full[i][:,1], full[i][:,0])
    #if processed is None:
    processed[:,:,i] = full[i][:30].astype(np.float64)
        #processed = full[i][:30].astype(np.float64)
    #else:
        #processed = np.vstack((processed, full[i][:30].astype(np.float64)))

#Different colours for each full
#Grids

plt.title('Raw Movement Traces')
plt.xlabel('Y coordinates')
plt.ylabel('X coordinates')

temp = None

for i in xrange(0,processed.shape[2]):
    if temp is None:
        temp = processed[:,:,i]
    else:
        temp = np.vstack((temp, processed[:,:,i]))

proc_mean = np.mean(temp,0)
proc_std = np.std(temp,0)

plt.figure()

for i in xrange(0,len(full)):
    plt.scatter(processed[:,0,i], processed[:,1,i], color=colors[i], s=60)


#plt.scatter(processed[:,0], processed[:,1], c='b', s=60)
plt.scatter(proc_mean[0], proc_mean[1], c='r', s=80)
plt.errorbar(proc_mean[0], proc_mean[1], xerr=proc_std[0], yerr=proc_std[1], fmt='o', ecolor='r', capthick=4)

"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

x = np.arange(10)
ys = [i+x+(i*x)**2 for i in range(10)]

colors = cm.rainbow(np.linspace(0, 1, len(ys)))
for y, c in zip(ys, colors):
    plt.scatter(x, y, color=c)
"""


plt.title('Scatter plots of starting poses, Experiment 1')
plt.xlabel('X coordinates')
plt.ylabel('Y coordinates')
plt.grid()