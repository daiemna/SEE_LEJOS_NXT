# -*- coding: utf-8 -*-
"""
Created on Sun Jun 07 17:51:43 2015

@author: Elladyr
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

"""
Starting poses
- forward
	- X-coord : -448.37492
	- Y-coord : 200.36568
	- Yaw : 0.59999

- Left
	- X-coord : -448.74010
	- Y-coord : 199.93731
	- Yaw : 0.58412
	
- Right
	- X-coord : -443.57052
	- Y-coord : 199.53851
	- Yaw : 0.63805
 
Steps
- Forward : 150
- Left : 160
- Right : 161 
 
Velocities
forward : 
- Linear: 6.67mm/timestep
- Angular : 0

left : 
- Linear : 6.25mm / timestep
- Angular : 0.0078 rad/timestep


right: 
- Linear : 6.21mm / timestep
- Angular : 0.0078 rad/timestep

"""

#0.59999 needs to be -0.05 for forward

linear_vel = 6.25
ang_vel = -0.0078

forward = np.array([[-448.74010, 199.93731, -0.05]])

theta = sp.Symbol('theta')
expr = sp.Matrix([[sp.cos(theta), 0], [sp.sin(theta), 0], [0, 1]])
rot = sp.lambdify(theta, expr, "numpy")

for i in xrange(1, 150):
    result = np.dot(rot(forward[i-1, 2]), [linear_vel, ang_vel]) + forward[i-1]
    forward = np.vstack((forward, result))
    
#plt.figure()
plt.plot(forward[:,0], forward[:,1], 'ro')