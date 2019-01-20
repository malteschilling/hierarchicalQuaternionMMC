'''
Created on 22.12.2018
@author: mschilling

Simulation of the Hierarchical Dual-Quaternion-based MMC network as a body model.
The task is a bimanual task - one arm is driven on the joint level (forward kinematics)
and the other arm is expected to follow this movement (inverse kinematic task) - both
arms should touch each other at the end effectors.

The internal body model consists of different layers. 
On the lower level there are dual-quaternion based MMC arm networks. 
These are coupled on the higher level through a MMC body network. 
In this case this is very simple as we only model the two arms and a connection between 
those. For the task of a coordinated movement, this connection was always set to 
zero length. 

Result is a plot showing distance between the end effectors of the left and right arm 
over time.
Distance increased during the movement up to 0.113 units, but this difference is not very
large as overall the mean distance between starting point and target point was 
2.021 (S.D. $\pm0.672$) units.

The left arm base was one unit length to the left from the right arm's shoulder. 
The body model had the task of moving the left arm where the right arm is. 
In the simulation, we selected all postures from the preceding simulation that were in 
reaching distance of the left arm.
The right arm followed in a random sequence a path through all remaining 561 postures. 
Again, the right arm moved for 25 iteration steps and afterwards the network got 
additional 25 iteration steps to even further settle into a stable state. 
The movement trajectories of the right arm were interpolated using SLERP on the joint 
level. 
Initially, the left arm was provided with the starting posture of the right arm. 
From there on the task for the left arm was to always point to the same point 
as the right arm.

'''
import math
import matplotlib.pylab as py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import numpy

import sys
sys.path.insert(0, '../1_SingleArm_DQ_ForwardKinematic/')
import pycgtypes.vec3 as vec3
import pycgtypes.quat as quat
from visualize_distance import DataVisualizerPlot

from numpy.random import normal

from dualQuaternionMmcArmNetwork import DualQuaternionMMCArmNetwork
from pycgtypes.quat import slerp as slerp

#######################################
#######################################
# GENERATE POSTURE AND PATH
# shuffling - and then these postures should 
# be subsequently visited
#######################################
#######################################

right_arm = DualQuaternionMMCArmNetwork()
vect_between_arms = numpy.array([1.,0,0])
##
# Variation 3 DoF first joint
dof_var_theta_0 = [0,0,0]
dof_var_theta_0[0] = [quat(0.05, [0,0,1]), quat(0.78, [0,0,1]), quat(1.57, [0,0,1])]
dof_var_theta_0[1] = [quat(0.05, [0,1,0]), quat(0.78, [0,1,0]), quat(1.57, [0,1,0])]
dof_var_theta_0[2] = [quat(0.05, [1,0,0]), quat(0.78, [1,0,0]), quat(-0.78, [1,0,0])]
theta_0_var = []
for first in dof_var_theta_0[0]:
    for second in dof_var_theta_0[1]:
        for third in dof_var_theta_0[2]:
            theta_0_var.append( (third * second * first) )
##
# second joint
theta_1_var = [quat(0.05, [0,0,1]), quat(0.78, [0,0,1]), quat(1.57, [0,0,1])]
##
# Variations 3 Dof third joint
dof_var_theta_2 = [0,0,0]
dof_var_theta_2[0] = [quat(0.05, [0,0,1]), quat(0.78, [0,0,1]), quat(1.57, [0,0,1])]
dof_var_theta_2[1] = [quat(0.05, [0,1,0]), quat(0.78, [0,1,0]), quat(-0.78, [0,1,0])]
theta_2_var = []
for first in dof_var_theta_2[0]:
    for second in dof_var_theta_2[1]:
        theta_2_var.append( (second * first) )
# Putting the postures together
all_rotation_series = []
for first in theta_0_var:
    for second in theta_1_var:
        for third in theta_2_var:
            all_rotation_series.append([first, second, third])
#all_rotation_series = [[quat(0.05, [0,0,1]),quat(0.05, [0,0,1]),quat(0.05, [0,0,1])],
#   [quat(0.05, [0,0,1]),quat(1.57, [0,0,1]),quat(1.57, [0,0,1])],
#   [quat(1.57, [0,0,1]),quat(1.55, [0,0,1]),quat(1.55, [0,0,1])]]
import random
# Creating random order = path to follow through the different postures
random.shuffle(all_rotation_series)
rotation_series = []
overall_dist = []
# Remove all postures that are unreachable for the left arm or to close 
for rot in all_rotation_series:
    target_vect = right_arm.forwardKinematics(rot)
    dist_vect = target_vect + vect_between_arms
    dist = numpy.sqrt( dist_vect[0]**2 + dist_vect[1]**2 + dist_vect[2]**2 )
    if dist < 3.:
        if dist > 0.5:
            rotation_series.append(rot)
            if len(rotation_series)>1:
                dist_vect = target_vect - old_vect
                overall_dist.append(numpy.sqrt( dist_vect[0]**2 + dist_vect[1]**2 + dist_vect[2]**2 ))
            else:
                old_vect = target_vect
overall_sum = sum(overall_dist)
overall_mean = numpy.mean(overall_dist)
overall_std = numpy.std(overall_dist)
print("Shuffled path through possible points: ", rotation_series[0], len(rotation_series), overall_sum, overall_mean, overall_std)

#rotation_series = [ [quat(1, [0,0,1]), quat(1, [0,0,1]), quat(1, [0,0,1])],
#   [quat(0.05, [0,0,1]), quat(1.57, [0,0,1]), quat(1.57, [0,0,1])],
#   [quat(1.05, [0,0,1]), quat(0.57, [0,0,1]), quat(2.57, [0,0,1])]]
    
# Setup data drawing collector
# Setup visualization class for distance plot in the end. 
distancePlotting = DataVisualizerPlot([1.,1.,1.]) #[1.3963, -0.5236])

#######################################
#######################################
# INIT Hierarchical Body Model
#######################################
#######################################
# Setting up lower level DQ arm models
left_arm = DualQuaternionMMCArmNetwork()
right_arm = DualQuaternionMMCArmNetwork()
from dualQuaternion import dualQuaternion

# Higher level is constituted simply as chain of vectors:
# - Between shoulders, 
# - vectors from shoulders towards end points of arms
# - connection between end effector = zero as those should touch each other in this task
vect_between_arms = numpy.array([1.,0,0])
# Arm vectors are already represented inside the lower level MMC networks.

i = 0
it_step = 51
number_targets = len(rotation_series)
data_norm_distance = [ [ [] for i in range(number_targets) ] for j in range(number_targets) ]
target_quat = [0,0,0]

#######################################
# Move right arm to starting posture
#######################################
for i in range(0,50):
    right_arm.theta_0[0].real_part = quat(rotation_series[0][0])
    right_arm.theta_1[0].real_part = quat(rotation_series[0][1])
    right_arm.theta_2[0].real_part = quat(rotation_series[0][2])
    right_arm.mmc_kinematic_iteration_step()

temp_vect = right_arm.get_target_coordinates()
target_mmc_right_vect = numpy.array([temp_vect[0][1], temp_vect[1][1],temp_vect[2][1]])

#######################################
# Move left arm to starting position
#######################################
# This constitutes in effect the higher level of the body model:
# a vector chain of the arm vectors and the connection vector between the shoulders
target_vect = vec3([(target_mmc_right_vect + vect_between_arms)[0],
    (target_mmc_right_vect + vect_between_arms)[1],
    (target_mmc_right_vect + vect_between_arms)[2]])
for i in range(0,50):
    left_arm.set_target_vect( target_vect )
#   left_arm.set_target_vect( target_mmc_right_vect + vect_between_arms )
#   left_arm.r[0].set_translation( target_mmc_right_vect + vect_between_arms)
#   left_arm.alpha[0].set_rotation(0,[0,0,1])
    left_arm.mmc_kinematic_iteration_step()

dist_both_tips = [ [] for j in range(it_step) ]

# Adapt path: change identical subsequent transformations
for target_posture in range(1,len(rotation_series)):
    if(rotation_series[target_posture - 1][0] == rotation_series[target_posture][0]):
        print("Adjusted equal quaternions")
        rotation_series[target_posture][0] = quat(rotation_series[target_posture][0].toAngleAxis()[0]*0.97, rotation_series[target_posture][0].toAngleAxis()[1])
    if(rotation_series[target_posture - 1][1] == rotation_series[target_posture][1]):
        print("Adjusted equal quaternions")
        rotation_series[target_posture][1] = quat(rotation_series[target_posture][1].toAngleAxis()[0]*0.97, rotation_series[target_posture][1].toAngleAxis()[1])
    if(rotation_series[target_posture - 1][2] == rotation_series[target_posture][2]):
        print("Adjusted equal quaternions")
        rotation_series[target_posture][2] = quat(rotation_series[target_posture][2].toAngleAxis()[0]*0.97, rotation_series[target_posture][2].toAngleAxis()[1])

#######################################
#######################################
# MOVEMENT SEQUENCE
# Make movement with right arm (25 iterations)
# and follow with left (same time + 25 it)
#######################################
#######################################
temp_vect = right_arm.get_manipulator_coordinates()
old_right_arm_end_point = numpy.array([temp_vect[0][3], temp_vect[1][3], temp_vect[2][3]])
movements_right_arm = []
# Iterate over the sequence of postures
for target_posture in range(1,len(rotation_series)):
    # Make a movement for 50 iterations
    for i in range(0,it_step) :
        # Active movement of right arm only for 25 iterations - using joint target values
        if i<26:
            target_quat[0] = slerp( (i*0.04), rotation_series[target_posture-1][0], rotation_series[target_posture][0])
            target_quat[1] = slerp( (i*0.04), rotation_series[target_posture-1][1], rotation_series[target_posture][1])
            target_quat[2] = slerp( (i*0.04), rotation_series[target_posture-1][2], rotation_series[target_posture][2])
        right_arm.theta_0[0].real_part = quat(target_quat[0])
        right_arm.theta_1[0].real_part = quat(target_quat[1])
        right_arm.theta_2[0].real_part = quat(target_quat[2])
        right_arm.mmc_kinematic_iteration_step()
        
        # Collect data of the movement
        if i<26:
            temp_vect = right_arm.get_manipulator_coordinates()
            right_arm_end_point = numpy.array([temp_vect[0][3], temp_vect[1][3], temp_vect[2][3]])
            diff_dist_right_vect =  old_right_arm_end_point - right_arm_end_point
            dist_right = numpy.sqrt( diff_dist_right_vect[0]**2 + diff_dist_right_vect[1]**2 + diff_dist_right_vect[2]**2 )
            movements_right_arm.append(dist_right)
            old_right_arm_end_point = right_arm_end_point
        
        temp_vect = right_arm.get_target_coordinates()
        target_mmc_right_vect = numpy.array([temp_vect[0][1], temp_vect[1][1],temp_vect[2][1]])
        
        # Take right arm position as input
        # and let left arm network follow
        target_vect = vec3([(target_mmc_right_vect + vect_between_arms)[0],
            (target_mmc_right_vect + vect_between_arms)[1],
            (target_mmc_right_vect + vect_between_arms)[2]])
        left_arm.set_target_vect( target_vect )
#        left_arm.r[0].set_translation( (target_mmc_right_vect + vect_between_arms ) )
#        left_arm.alpha[0].set_rotation(0,[0,0,1])
        left_arm.mmc_kinematic_iteration_step()
        
        temp_vect = left_arm.get_target_coordinates()
        target_mmc_left_vect = numpy.array([temp_vect[0][1], temp_vect[1][1],temp_vect[2][1]])
        
        diff_vect = target_mmc_left_vect - (target_mmc_right_vect +vect_between_arms)
        dist = numpy.sqrt( diff_vect[0]**2 + diff_vect[1]**2 + diff_vect[2]**2 )
        dist_both_tips[i].append(dist)
        
    print('### Movement: ', target_posture, dist_both_tips[25][-1])

#######################################
#######################################
# ANALYSIS of movement over time
#######################################
#######################################
stat_mean = [ [] for j in range(it_step) ]
stat_stdev = [ [] for j in range(it_step) ]
for time_steps in range(0,it_step):
    stat_mean[time_steps] = numpy.mean(dist_both_tips[time_steps])
    stat_stdev[time_steps] = numpy.std(dist_both_tips[time_steps])
stat_lower_limit = [mv - sd for mv, sd in zip(stat_mean, stat_stdev)] 
stat_upper_limit = [mv + sd for mv, sd in zip(stat_mean, stat_stdev)] 
print("## STATISTICS ####")
print("Shuffled path through possible points: ", rotation_series[0], len(rotation_series), overall_sum, overall_mean, overall_std)
print("  Overall mean distance between fingertipps: ", numpy.mean(stat_mean), " at end of movement: ", stat_mean[25], " / 50th: ", stat_mean[50])
print("  Overall standard deviation between fingertipps: ", numpy.mean(stat_stdev), "at end of movement: ", stat_stdev[25], " / 50th: ", stat_stdev[50])
print("  Overall movement right arm during one iteration (first 26 steps): ", numpy.mean(movements_right_arm), numpy.std(movements_right_arm))
print(stat_mean)
print(stat_stdev)

distancePlotting.draw_distance_fingertips_curve(stat_mean, stat_lower_limit, stat_upper_limit, "distanceBoth.pdf")
raw_input("Wait")