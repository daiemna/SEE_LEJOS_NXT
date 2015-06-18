# -*- coding: utf-8 -*-
"""
Created on Mon May 11 15:30:44 2015

@author: Elladyr
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import matplotlib.cm as cm
import sympy as sp

#cams = AP.list_cameras() #lifecam19, lifecam20 and lifecam22
#img = AP.get_camera_image(cams[1])
#plt.imshow(image[0][:,:,0])
#For experiment, we are using lifecam19
idx = 1

fpath = 'E:/Studies/Semester 4/Scientific Experimentation and Evaluation/Lecture 5/Youbot experiment/'
fname = '/marker_22203_exp_'

folders = ['Testing_150 timesteps/', 'Testing_160 timesteps/', 'Testing_161 timesteps/']

array = None
full = []

"""
Indices
0 - Timestamp
1 - X coord
2 - Y coord
3 - Z coord
4 - Roll
5 - Pitch
6 - Yaw
"""

for folder in folders:
    for name in sorted(glob.glob(fpath+folder+'*.txt')):
        with open(name) as f:
            content = f.read().splitlines()
            f.close()
        for i in xrange(0, len(content)):
            temp = np.array(content[i].split())
    
            if array is None:
                array = np.asarray(temp[[0,3,4,5,6,7,8]]) #Pick only the timestamp and rest of the data
            else:
                array = np.vstack((array, np.asarray(temp[[0,3,4,5,6,7,8]])))
    
        full.append(array)
        array = None

colors = plt.cm.Set1(np.linspace(0, 1, 21))

processed = None
#processed = np.zeros((len(full[0]),7,len(full)))

for i in xrange(0, len(full)):
    plt.plot(full[i][:,1], full[i][:,2], label=('Run ' + str(i+1)))
    
    if processed is None:
        processed = full[i].astype(np.float64)
    else:
        processed = np.vstack((processed, full[i].astype(np.float64)))

"""
Indices
0 - 3149 : Forward Exp
3150 - 6509 : Left Exp
6510 - 9729 : Right exp
"""

"""
Extraction of velocities
"""

#total_vel = None

fwd = processed[:3150,[1,2,6]]
cmd_fwd = np.zeros((3129,2)) #149 * 21 (150 - 1)
cmd_fwd[:,0] = 6.67

mea_fwd = None

for idx in xrange(0, len(fwd), 150):
    fwd_vel = fwd[idx:idx+150][1:,:] - fwd[idx:idx+150][:-1,:]
    if mea_fwd is None:
        mea_fwd = fwd_vel
    else:
        mea_fwd = np.vstack((mea_fwd, fwd_vel))

left = processed[3150:6510, [1,2,6]]
cmd_left = np.zeros((3339,2)) #159 * 21 (160 - 1)
cmd_left[:,0] = 6.25
cmd_left[:,1] = -0.0078

mea_left = None

for idx in xrange(0, len(left),160):
    left_vel = left[idx:idx+160][1:,:] - left[idx:idx+160][:-1,:]
    if mea_left is None:
        mea_left = left_vel
    else:
        mea_left = np.vstack((mea_left, left_vel))

right = processed[6510:9730,[1,2,6]]
cmd_right = np.zeros((3200,2)) #160 * 20 (161 - 1)
cmd_right[:,0] = 6.21
cmd_right[:,1] = 0.0078

mea_right = None

for idx in xrange(0, len(right),161):
    right_vel = right[idx:idx+161][1:,:] - right[idx:idx+161][:-1,:]
    if mea_right is None:
        mea_right = right_vel
    else:
        mea_right = np.vstack((mea_right, right_vel))

total_vel = np.vstack((mea_fwd, mea_left, mea_right))
cmd_all = np.vstack((cmd_fwd, cmd_left, cmd_right))


"""
plt.figure()
for i in xrange(0, len(full)):
    plt.plot(full[i][:,1], label=('X-coord only, Run ' + str(i+1)))
"""

#Marker offset between marker frame and robot frame

"""
#Forward movement parameters
linear_vel = 6.67
ang_vel = 0
#trajectory = np.array([[-448.37492, 200.36568, 0.59999]])
trajectory = np.array([[-448.37492, 200.36568, -0.07]]) #Offset : -0.66999
"""
"""
#Left parameters
linear_vel = 6.25
ang_vel = -0.0078
#trajectory = np.array([[-443.57052, 199.93731, 0.58412]])
trajectory = np.array([[-443.57052, 199.93731, -0.08578]]) #Offset : -0.66999
"""
"""
#Right movement parameters
linear_vel = 6.21
ang_vel = 0.0078
#trajectory = np.array([[-448.74010, 199.53851, 0.63805]])
trajectory = np.array([[-448.74010, 199.53851, -0.13]]) #Offset : -0.66999
"""
"""
theta = sp.Symbol('theta')
expr = sp.Matrix([[sp.cos(theta), 0], [sp.sin(theta), 0], [0, 1]])
rot = sp.lambdify(theta, expr, "numpy")

for i in xrange(1, len(full[0])):
    result = np.dot(rot(trajectory[i-1, 2]), [linear_vel, ang_vel]) + trajectory[i-1]
    trajectory = np.vstack((trajectory, result))
    
#plt.figure()
plt.plot(trajectory[:,0], trajectory[:,1], 'ro', ms=8, label='Commanded trajectory')


#Different colours for each full
#Grids
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2,ncol=7, mode="expand", borderaxespad=1.)
plt.title('Raw Movement Traces')
plt.xlabel('X-coordinates')
plt.ylabel('Y-coordinates')
"""

"""
Extraction of velocities
"""



theta = sp.Symbol('theta')
expr = sp.Matrix([[sp.cos(theta), 0], [sp.sin(theta), 0], [0, 1]])
rot = sp.lambdify(theta, expr, "numpy")

rot_inv = sp.lambdify(theta, expr.T, "numpy")

#Forward
wld_fwd = np.zeros((len(mea_fwd),2))
for i in xrange(0, len(mea_fwd)):
    wld_fwd[i] = np.dot(rot_inv(-0.07), mea_fwd[i,:])

#Left
wld_left = np.zeros((len(mea_left),2))
for i in xrange(0, len(mea_left)):
    wld_left[i] = np.dot(rot_inv(-0.08578), mea_left[i,:])

#Right
wld_right = np.zeros((len(mea_right),2))
for i in xrange(0, len(mea_right)):
    wld_right[i] = np.dot(rot_inv(-0.13), mea_right[i,:])

wld_vel = np.vstack((wld_fwd, wld_left, wld_right))

import scipy.optimize as scop

def func(x, k, omega):
    return k*x + omega

params, pcov = scop.curve_fit(func, cmd_all[:,0], wld_vel[:,0])
params_omega, pcov_omega = scop.curve_fit(func, cmd_all[:,1], wld_vel[:,1])


#avg_measured = np.mean(processed, axis=2) #(150, 7)

#from sklearn import linear_model
#clf = linear_model.LinearRegression()
#clf.fit()

#Starting pose point plots
"""
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
    plt.scatter(processed[:,1,i], processed[:,2,i], color=colors[i], s=60, label=('Run ' + str(i+1)))

plt.scatter(proc_mean[1], proc_mean[2], c='r', s=80)
plt.errorbar(proc_mean[1], proc_mean[2], xerr=proc_std[1], yerr=proc_std[2], fmt='o', ecolor='r', capthick=4)

plt.title('Scatter plots of starting poses, Experiment Turning Right')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2,ncol=7, mode="expand", borderaxespad=1.)
plt.xlabel('X coordinates')
plt.ylabel('Y coordinates')
plt.grid()
"""

#For processed data only
"""
timesteps = None

for i in xrange(0, len(full)):
    if timesteps is None:
        timesteps = np.asarray(full[i][1:,0], dtype=np.float64) - np.asarray(full[i][:-1,0], dtype=np.float64)
    else:
        timesteps = np.vstack((timesteps, (np.asarray(full[i][1:,0], dtype=np.float64) - np.asarray(full[i][:-1,0], dtype=np.float64))))


average_timestep = np.mean(timesteps)
total_time = np.mean(timesteps) * len(full[0])

print 'Average per timestep : ', average_timestep
print 'Number of timesteps : ', len(full[0])
print 'Total time for trajectory : ', total_time
"""


"""
rot = np.array([[1.1924880638503051e-08, 9.9999999999999978e-01, -4.5000000536619623e+01], 
                [9.9999999999999978e-01, -1.1924880638503051e-08, -4.4999999463380362e+01],
                [0,0,1]])

plt.figure()

for j in xrange(0, len(full)):
    test = full[j][:,0:3].astype(np.float64)
    #plt.plot(test[:,1], test[:,0])
    final = np.dot(rot,test.T)
    plt.plot(final[:,0], final[:,1], label=('Run ' + str(j+1)))
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2,ncol=7, mode="expand", borderaxespad=1.)

plt.title('After applying rotation matrix')
plt.xlabel('X-coordinates')
plt.ylabel('Y-coordinates')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2,ncol=7, mode="expand", borderaxespad=1.)
"""

print 'Done'