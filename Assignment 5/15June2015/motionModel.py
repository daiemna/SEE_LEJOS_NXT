# -*- coding: utf-8 -*-
"""
Created on Mon May 11 15:30:44 2015

@author: Elladyr,haramoz
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import matplotlib.cm as cm
import sympy as sp
import random
#cams = AP.list_cameras() #lifecam19, lifecam20 and lifecam22
#img = AP.get_camera_image(cams[1])
#plt.imshow(image[0][:,:,0])
#For experiment, we are using lifecam19
idx = 1

fpath = '/home/arkid/Documents/Third Semester/SEE/day6/see_exp_readings/processed_transformed_seven_values/'
folders = ['processed_forward/', 'processed_left/', 'processed_right/']

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
            #print name
            content = f.read().splitlines()
            f.close()
        for i in xrange(0, len(content)):
            temp = np.array(content[i].split())
            #print temp[[0,1,2,3,4,5]]
            if array is None:
                array = np.asarray(temp[[0,1,2,3,4,5,6]]) #Pick only the timestamp and rest of the data
            else:
                array = np.vstack((array, np.asarray(temp[[0,1,2,3,4,5,6]])))
    
        full.append(array)
        array = None

colors = plt.cm.Set1(np.linspace(0, 1, 21))

processed = None
#processed = np.zeros((len(full[0]),7,len(full)))

for i in xrange(0, len(full)):
    #plt.plot(full[i][:,1], full[i][:,2], label=('Run ' + str(i+1)))
    
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

def func(x, k, b):
    return k*x + b

params, pcov = scop.curve_fit(func, cmd_all[:,0], wld_vel[:,0])
params_w, pcov_w = scop.curve_fit(func, cmd_all[:,1], wld_vel[:,1])

print 'K_v and b_v ', params
print 'K_w and b_w ', params_w

"""
k_v = 3.7603808
b_v = -18.58159586

k_omega = 0.50574324
b_omega = -0.00545543
"""

v_eff = params[0] * cmd_all[:,0] + params[1]
w_eff = params_w[0] * cmd_all[:,1] + params_w[1]

import scipy.spatial.distance as sp_dist

def opt_alpha(alpha):
    e1 = alpha[0]*v_eff + alpha[1]*w_eff
    e2 = alpha[2]*v_eff + alpha[3]*w_eff
    
    v = v_eff + np.random.normal(e1)
    w = w_eff + np.random.normal(e2)
    
    all_v = np.vstack((v, w))
    
    cov_mat = np.cov(wld_vel.T)
    
    total_dist = 0.0
    for i in xrange(0, len(all_v.T)):
        total_dist += sp_dist.mahalanobis(all_v.T[i], wld_vel[i], np.linalg.inv(cov_mat))
    
    return total_dist
    
#opt_alpha(np.array([1.0, 0.0, 1.0, 0.0]))
no_of_trials = 100
distances = []
bestAlphas = []
filehandler =  open('Exhaustive_search_results'+'.txt', 'w')
for j in range(no_of_trials):
    alp = []
    #for i in range(4):
    alp= np.random.uniform(0,1,4)
    print "Initial guess for alphas:", alp
    outputs = scop.fmin(opt_alpha, np.array(alp), maxiter=40, full_output=1, retall=1)

    alphas = outputs[0]
    min_dist = outputs[1]
    print min_dist
    distances.append(min_dist)
    if min_dist == np.min(distances):
        bestAlphas.append(alphas)
        soln_iter = None
        for i in xrange(0, len(outputs[5])):
            if soln_iter is None:
                soln_iter = outputs[5][i]
            else:
                soln_iter = np.vstack((soln_iter, outputs[5][i]))
    line = str(j) + ' Alphas ' + str(alp) + ' min_dist ' + str(min_dist) + '\n'
    filehandler.write(line)
filehandler.close()
print "Best Alphas:", bestAlphas
print "min distance:", round(np.min(distances),3)



plt.figure()
plt.title("best alphas")
plt.plot(soln_iter[:,0], label='Alpha 1')
plt.plot(soln_iter[:,1], label='Alpha 2')
plt.plot(soln_iter[:,2], label='Alpha 3')
plt.plot(soln_iter[:,3], label='Alpha 4')
plt.legend()
plt.show()
print 'Done'