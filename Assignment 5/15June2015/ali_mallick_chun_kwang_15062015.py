#!/usr/bin/env python
# Software License Agreement (BSD License)
#
# Copyright (c) 2012, Willow Garage, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of Willow Garage, Inc. nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Original Authors: Steven Gray, Christian Dornhege, Georg Bartels, Jihoon Lee, John Schulmann
#                   Team 1, PR2 Workshop, Freiburg, Germany
# Edits: Tommaso Cavallari

"""
Adapted from http://www.scipy.org/Cookbook/Least_Squares_Circle
Input: points to fit to a circle, (x[], y[])
Output: circle center, 
"""
'''__Authors__:Elladyr,haramoz'''
from numpy import *
import random
from scipy import optimize, linalg
from mpl_toolkits.mplot3d import Axes3D
import math
from matplotlib import pyplot as p, cm, colors # only needed if want to plot separately
import copy
import csv

class circleFit:
    def __init__(self,minimumNoOfPoses):
        self.minimumNoOfPoses = minimumNoOfPoses
        self.xValues = []
        self.yValues = []
        self.zValues = []
        self.listOfPoses = []
        self.center = None
        self.axis = None
        self.radius = None
        self.speed = None
        #self.path = path
        self.full = []
        self.x_in = []
        self.y_in = []
        self.z_in = []
        self.time = []
        self.roll = []
        self.pitch = []
        self.yaw = []
        self.total_x_in = []
        self.total_y_in = []
        self.total_z_in = []
        self.circlefitResults = []
        self.legend_flag = True

    def readexperimentData(self,expName):
        idx = 1

        #fpath = '/home/arkid/Documents/Third Semester/SEE/day6/see_exp_readings/forward_exp/total'
        #fpath ='/home/arkid/Documents/Third Semester/SEE/day6/see_exp_readings/'+expName+'/total'
        fpath ='/home/arkid/Documents/Third Semester/SEE/day6/see_exp_readings/'+expName
        fname = '/marker_22203_exp_'

        array = None
        self.x_in = []
        self.y_in = []
        self.z_in = []
        self.time = []
        self.roll = []
        self.pitch = []
        self.yaw = []
        #print output
        first_pose_x = []
        first_pose_y = []
        last_pose = []
        colors = p.cm.Set1(linspace(0, 1, 21))

        for idx in xrange(1,22):
            #self.x_in = []
            #self.y_in = []
            #self.z_in = []

            a = str(fpath+fname+str(idx)+'.txt')
            
            with open(a) as f:
                content = f.read().splitlines()
                f.close()
                
                first_pose_collected = False
            for i in xrange(0, len(content)):
                temp = content[i].split()
                #print temp
                if array is None:
                    array = asarray(temp[3:])
                else:
                    array = vstack((array, asarray(temp[3:])))
                self.x_in.append(float(temp[3]))
                self.y_in.append(float(temp[4]))
                self.z_in.append(float(temp[5]))
                self.time.append(float(temp[0]))
                self.roll.append(float(temp[6]))
                self.pitch.append(float(temp[7]))
                self.yaw.append(float(temp[8]))
                self.total_x_in.append(float(temp[3]))
                self.total_y_in.append(float(temp[4]))
                self.total_z_in.append(float(temp[5]))
                if not first_pose_collected:
                    first_pose_x.append(float(temp[3]))
                    first_pose_y.append(float(temp[4]))
                    first_pose_collected = True
            #self.planarVisualization(idx)
            #self.find_circle_posestamped(self.x_in,self.y_in,self.z_in)
            self.full.append(array)
            array = None
            #print self.x_in
        '''print expName
        print "first_pose_x", std(first_pose_x),mean(first_pose_x)
        print "first_pose_y",std(first_pose_y),mean(first_pose_y)
        #print len(first_pose_x)

        p.figure()
        #p.title(expName)
        p.grid()
        p.xlabel("x-coordinates(mm)")
        p.ylabel("y-coordinates(mm)")
        for i in xrange(0,21):
            p.scatter(first_pose_x[i],first_pose_y[i],color=colors[i], s=60, label=('Run ' + str(i+1)))
        p.legend(bbox_to_anchor=(0.0, 1.02, 1., .102),ncol=7, borderaxespad=1.0,title=expName,mode='expand',scatterpoints=1)
        #p.show()
        #p.figure()'''
    '''def calc_R(self, xc, yc, x, y):
        """ Calculate the distance of each 3D point from the center (xc, yc). """
        return sqrt((x-xc)**2 + (y-yc)**2)

    def f_2(self, c, x_in, y_in):
        """ Calculate the algebraic distance between the 3D points and the mean circle centered at c=(xc, yc). """
        x_c, y_c = c
        Ri = self.calc_R(x_c, y_c, x_in, y_in)
        return Ri - Ri.mean()'''

    def fit_plane(self, x_in, y_in, z_in):
        """ 
        Fit a plane through the input points using Principal Component Analysis. 
        Args:
            x_in: the x coordinates of the points
            y_in: the y coordinates of the points
            z_in: the z coordinates of the points
        Returns:
            an array containing the 4 plane coefficients
        """
        points = array([x_in, y_in, z_in])
        center = points.mean(axis=1)
        points[0,:] -= center[0]
        points[1,:] -= center[1]
        points[2,:] -= center[2]
        covariance  = cov(points)
        
        eval, evec  = linalg.eig(covariance)
        ax_id = argmin(eval)
        plane_normal = evec[:, ax_id]
        plane_d = dot(center.T, plane_normal)
        
    #    print "center: %s" % center
    #    print "cov: %s" % covariance
    #    print "eval: %s" % eval
    #    print "evec: %s" % evec
    #    print "axe: %s" % ax_id
    #    print "normal: %s" % plane_normal
    #    print "plane_d: %s" % plane_d
        
        return array([plane_normal[0], plane_normal[1], plane_normal[2], plane_d])
    
    def project_points_to_plane(self, x_in, y_in, z_in, plane_coeffs):
        """ 
        Project a set of points on a plane defined by its coefficients.
        Args:
            x_in: the x coordinates of the points
            y_in: the y coordinates of the points
            z_in: the z coordinates of the points
            plane_coeffs: the 4 plane coefficients
        Returns:
            proj_x: the projected x coordinates
            proj_y: the projected y coordinates
            proj_z: the projected z coordinates
            origin: the origin (on the plane) as the point pointed by the plane_coeffs vector
        """
        # define the origin (on the plane) as the point pointed by the coeff vector
        # d * (a, b, c)
        origin = (plane_coeffs[3]) * plane_coeffs[0:3]
    #    print "origin: %s" % origin
        
        # for each point project and obtain its x, y coords
        v_x = x_in - origin[0]
        v_y = y_in - origin[1]
        v_z = z_in - origin[2]
        
        dist = v_x * plane_coeffs[0] + v_y * plane_coeffs[1] + v_z * plane_coeffs[2]
    #    print "dist: %s" % dist
        
        proj_x = x_in - dist * plane_coeffs[0]
        proj_y = y_in - dist * plane_coeffs[1]
        proj_z = z_in - dist * plane_coeffs[2]
        
        return proj_x, proj_y, proj_z, origin   

    def points3d_to_2d(self, x_in, y_in, z_in, plane_coeffs):
        """
        Project 3D points on a plane and return their 2D coordinates and the two axii defining the coordinates.
        Args:
            x_in: the x coordinates of the points
            y_in: the y coordinates of the points
            z_in: the z coordinates of the points
            plane_coeffs: the 4 plane coefficients
        Returns:
            x_proj: the projected x coordinates (in the x_axis direction)
            y_proj: the projected y coordinates (in the y axis direction)
            x_axis: the determined x axis
            y_axis: the determined y axis
        """
        # define arbitrary axis
        # the origin is defined by the plane_coeffs
        x_axis = None
        y_axis = None
        
        if array_equal(plane_coeffs[0:3], array([1,0,0])):
            x_axis = cross(plane_coeffs[0:3], array([0,1,0]))
        else:
            x_axis = cross(plane_coeffs[0:3], array([1,0,0]))
            
        y_axis = cross(plane_coeffs[0:3], x_axis)
        
        x_proj = []
        y_proj = []
        
        for i in range(len(x_in)):
            point = array([ x_in[i], y_in[i], z_in[i] ])
            x_proj.append(dot(point, x_axis))
            y_proj.append(dot(point, y_axis))
            
        return x_proj, y_proj, x_axis, y_axis

    def calc_R(self, xc, yc, x, y):
        """ Calculate the distance of each 3D point from the center (xc, yc). """
        return sqrt((x-xc)**2 + (y-yc)**2)

    def f_2(self, c, x_in, y_in):
        """ Calculate the algebraic distance between the 3D points and the mean circle centered at c=(xc, yc). """
        x_c, y_c = c
        Ri = self.calc_R(x_c, y_c, x_in, y_in)
        return Ri - Ri.mean()


    def find_circle(self, x_in, y_in, plot_circle=True):
        """
        Find a the best circle that passes through the input points. Computes also the angular speed.
        Args:
            x_in: the x coordinates of the points
            y_in: the y_coordinates of the points
            times: the observation time for each couple of x-y values
            plot_circle: an optional arguments that allows for a graphical visualization of the results
        Returns:
            xc_2: the x coordinate of the center
            yc_2: the y coordinate of the center
            R_2: the circle radius
            ang_vel: the angular rotation speed
        """
        # coordinates of the barycenter
        x_m = mean(x_in)
        y_m = mean(y_in)
    
        center_estimate = x_m, y_m
        center_2, ier = optimize.leastsq(self.f_2, center_estimate, args=(x_in, y_in))
    
        xc_2, yc_2 = center_2
        Ri_2       = self.calc_R(xc_2, yc_2, x_in, y_in)
        R_2        = Ri_2.mean()
        
            
        self.circlefitResults.append([xc_2, yc_2,R_2,x_in,y_in])
        '''if plot_circle:
            self.plot_all(xc_2, yc_2, R_2, x_in, y_in)'''
    
        return xc_2, yc_2, R_2

    def find_circle_posestamped(self, x_in,y_in,z_in):
        """
        Given an EstimateRotationRequest containing a list of object poses compute if possible the rotation parameters.
        Args:
            req: an EstimateRotationRequest containing a list of PoseWithCovarianceStamped.
        Returns:
            response: an EstimateRotationResponse containing the rotation parameters for the rotating object.
        """
        '''pose_stamped_list = posestampList
        
        if len(pose_stamped_list) < self.minimumNoOfPoses:
            print 'Not enough poses to estimate a rotation'
            #return EstimateRotationResponse(success=False)
            return None'''
        
        x_in = x_in #array('f')
        y_in = y_in #array('f')
        z_in = z_in #array('f')
        times_in = []

            
        # 1st thing: find the supporting plane
        plane_coeffs = self.fit_plane(x_in, y_in, z_in)
        #print "plane coeffs: %s" % plane_coeffs
        
        # 2nd thing: project the points on the plane and find their 2d coords wrt the "origin" point on the plane in an arbitrary reference frame
        proj_x, proj_y, proj_z, origin = self.project_points_to_plane(x_in, y_in, z_in, plane_coeffs)
        x_proj2d, y_proj2d, x_axis, y_axis = self.points3d_to_2d(proj_x, proj_y, proj_z, plane_coeffs)
        #print x_axis, y_axis
        #writeFeatures(self,xlist,ylist,filename)
        # 3rd: now find the circle.
        c_x, c_y, self.radius = self.find_circle(x_proj2d, y_proj2d, True)
        print "Estimated center:",c_x, c_y,"Estimated radius:", self.radius
        # c_x and c_y are relative to the origin on the plane, convert them back to world coords
        c_vector = origin + x_axis * c_x + y_axis * c_y
        print c_vector      
        return c_x, c_y, self.radius

    # plotting functions
    def plot_all(self, xc_2, yc_2, R_2, x, y):
        #print "going to display"
        #p.close('all')
        """ 
        Draw data points, best fit circles and center for the three methods,
        and adds the iso contours corresponding to the fiel residu or residu2
        """
    
        #f = p.figure( facecolor='white')  #figsize=(7, 5.4), dpi=72,
        p.axis('equal')
    
        speed_fit = linspace(-pi, pi, 180)
    
        x_fit2 = xc_2 + R_2*cos(speed_fit)
        y_fit2 = yc_2 + R_2*sin(speed_fit)
        p.plot(x_fit2, y_fit2, 'k--', label="leastsq", lw=2)
    
        # draw
        p.xlabel('x-coordinates(mm)')
        p.ylabel('y-coordinates(mm)')
    
        p.draw()
        xmin, xmax = p.xlim()
        ymin, ymax = p.ylim()
    
        vmin = min(xmin, ymin)
        vmax = max(xmax, ymax)
    
        # plot input data
        p.plot(x, y, 'ro', label='data',)
        if self.legend_flag:
            p.legend(loc='best',labelspacing=0.1 )
            self.legend_flag = False
        #p.xlim(xmin=vmin, xmax=vmax)
        #p.ylim(ymin=vmin, ymax=vmax)
    
        p.grid()
        p.title('Least Squares Circle')
       
        #p.show()

    def writeFeaturesOld(self,time, proj_x, proj_y, proj_z, roll, pitch, yaw,fileName):
        xlist = copy.deepcopy(proj_x)
        ylist = copy.deepcopy(proj_y)
        
        path = './'

        with open(path+fileName+'_transformed'+'.csv', 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            eachLine = []
            for i in range(len(xlist)): 
                eachLine.append(time[i])
                eachLine.append(xlist[i])
                eachLine.append(ylist[i])
                eachLine.append(roll[i])
                eachLine.append(pitch[i])
                eachLine.append(yaw[i])
                writer.writerow(eachLine)
                eachLine = []

    def writeFeatures(self,time, proj_x, proj_y, proj_z, roll, pitch, yaw,fileName):
        xlist = copy.deepcopy(proj_x)
        ylist = copy.deepcopy(proj_y)
        zlist = copy.deepcopy(proj_z)
        
        path = './'

        filehandler =  open(path+fileName+'_transformed'+'.txt', 'w')
        eachLine = []
        for i in range(len(xlist)): 
            eachLine.append(time[i])
            eachLine.append(xlist[i])
            eachLine.append(ylist[i])
            eachLine.append(zlist[i])
            eachLine.append(roll[i])
            eachLine.append(pitch[i])
            eachLine.append(yaw[i])
            line = str(time[i])+' '+str(xlist[i])+' '+str(ylist[i])+' '+str(zlist[i])+' '+str(roll[i])+' '+str(pitch[i])+' '+str(yaw[i])+'\n'
            filehandler.write(line)
            eachLine = []

    def outlierRemoval(self,x_proj2d, y_proj2d):
        x_proj2d = copy.deepcopy(x_proj2d)
        y_proj2d = copy.deepcopy(y_proj2d)
        outliers = []
        #print x_proj2d[455]
        #print len(x_proj2d),len(y_proj2d)
        for j in range(len(x_proj2d)):
            #print x_proj2d[j],j
            if x_proj2d[j] > -1210:
                if y_proj2d[j] < -1680:
                    outliers.append((x_proj2d[j],y_proj2d[j]))
                    del x_proj2d[j]
                    del y_proj2d[j]
        return x_proj2d, y_proj2d

if __name__ == "__main__":

    circleFitObj = circleFit(15)
    #To read the poses from the rosbag files
    #listofPoseStamped = circleFitObj.readRealPoses(paths[index])
    #exp_names = ['right_exp','left_exp','forward_exp']
    exp_names = ['Processed_right','Processed_left','Processed_forward']
    arc_exp_names = ['Processed_right','Processed_left','Processed_forward']
    #arc_exp_names = ['right_exp','left_exp','forward_exp']
    for exp_name in exp_names:
        circleFitObj.readexperimentData(exp_name)
    #Setting the input values 
    x_in = circleFitObj.total_x_in
    y_in = circleFitObj.total_y_in
    z_in = circleFitObj.total_z_in
    #Fitting the plane
    plane_coeffs = circleFitObj.fit_plane(x_in, y_in, z_in)
    #Project the points to the plane
    proj_x, proj_y, proj_z, origin = circleFitObj.project_points_to_plane(x_in, y_in, z_in, plane_coeffs)
    #Convert from 3D to 2D
    x_proj2d, y_proj2d, x_axis, y_axis = circleFitObj.points3d_to_2d(proj_x, proj_y, proj_z, plane_coeffs)
    
    p.xlabel('x-coordinates(mm)')
    p.ylabel('y-coordinates(mm)')
    p.title('Observed robot trajectory')
    p.plot(x_proj2d, y_proj2d,"go",alpha=0.6)

    for exp_name in arc_exp_names:
        circleFitObj.readexperimentData(exp_name)
        x_in = circleFitObj.x_in
        y_in = circleFitObj.y_in
        z_in = circleFitObj.z_in
        time = circleFitObj.time
        roll = circleFitObj.roll
        pitch = circleFitObj.pitch
        yaw = circleFitObj.yaw

        proj_x, proj_y, proj_z, origin = circleFitObj.project_points_to_plane(x_in, y_in, z_in, plane_coeffs)
        x_proj2d, y_proj2d, x_axis, y_axis = circleFitObj.points3d_to_2d(proj_x, proj_y, proj_z, plane_coeffs)
        #This projected points are needed for the test.
        circleFitObj.writeFeatures(time, proj_x, proj_y, proj_z, roll, pitch, yaw, exp_name)
        #circleFitObj.outlierRemoval(x_proj2d, y_proj2d)
        c_x, c_y, radius = circleFitObj.find_circle(x_proj2d, y_proj2d, True)
        print exp_name,'Radius:',radius 
        #circleFitObj.plot_all(c_x, c_y, radius,x_proj2d, y_proj2d)
        #p.plot(c_x, c_y,"mo")

    #p.show()