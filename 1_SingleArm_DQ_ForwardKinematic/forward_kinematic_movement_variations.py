'''
Forward kinematic simulation series:

Test convergence of a dual-quaternion based Mean of Multiple Computation network
in a series of forward kinematic tasks for a three segmented arm (7 DoFs).

A single human-like arm was used and controlled through given joint positions. 
The simulated arm consisted of three segments, all were assumed to be unit length. 
Overall, the arm had seven DoFs (like a human arm). 
The task for the internal MMC arm model was to solve the forward kinematic problem. 

Given a joint configuration, the task for the internal model was to estimate the current 
position in space of the tip of the arm. 
We tested the internal model on a large number of movements between different postures. 
In each simulation run the internal model was driven from a starting posture to one of 
two end postures: On the one hand, a fully stretched out arm, leading for many starting 
postures to large movements across the workspace. 
On the other hand, a completely folded arm.

The movements were controlled on the joint level. For each joint the start and end joint 
rotation was given and the movement had to be completed in 25 simulation steps. 
The in between joint rotations were obtained through interpolation using 
Spherical Linear Interpolation (SLERP) of the dual quaternions representing the rotation 
angle and axis. In this way all the joints were driven for 25 iteration steps and the 
movement followed the shortest path. 
During the movement (and for an additional 25 time steps afterwards) we recorded the 
internal arm models estimate of the end position.

We defined a set of postures by variation of the degrees of freedom.
See supplemental material for a figure and description of the postures.

@author: mschilling
'''
import math
import numpy
from numpy.random import normal

import pycgtypes.vec3 as vec3
import pycgtypes.quat as quat

from visualize_distance import DataVisualizerPlot

#angle_series = [numpy.array([1.513, 0.23, -2.38]), numpy.array([1.513, 0.67, -2.38])]

#######################################
# Defining the target postures
# Variation of the seven degrees of freedom
#######################################
'''
The different postures and joint representations for the joints were derived through 
sequential concatenation of individual transformations. 

For the first joint there are three DoFs and for each of those dimensions we used 
three different rotational positions.
For the elbow joint there is only one degree of freedom and for the third joint 
we only looked at two DoFs as we are not interested in the orientation of the manipulator 
in this simulation 

This results in 729 postures and, importantly, movements starting from these postures 
cover the whole working range.
'''
dof_var_theta_0 = [0,0,0]
dof_var_theta_0[0] = [quat(0.05, [0,0,1]), quat(0.78, [0,0,1]), quat(1.57, [0,0,1])]
dof_var_theta_0[1] = [quat(0.05, [0,1,0]), quat(0.78, [0,1,0]), quat(1.57, [0,1,0])]
dof_var_theta_0[2] = [quat(0.05, [1,0,0]), quat(0.78, [1,0,0]), quat(-0.78, [1,0,0])]
theta_0_var = []
# Combining them to one joint
for first in dof_var_theta_0[0]:
	for second in dof_var_theta_0[1]:
		for third in dof_var_theta_0[2]:
			theta_0_var.append( (third * second * first) )
# Second joint
theta_1_var = [quat(0.05, [0,0,1]), quat(0.78, [0,0,1]), quat(1.57, [0,0,1])]
# Third joint
dof_var_theta_2 = [0,0,0]
dof_var_theta_2[0] = [quat(0.05, [0,0,1]), quat(0.78, [0,0,1]), quat(1.57, [0,0,1])]
dof_var_theta_2[1] = [quat(0.05, [0,1,0]), quat(0.78, [0,1,0]), quat(-0.78, [0,1,0])]
theta_2_var = []
for first in dof_var_theta_2[0]:
	for second in dof_var_theta_2[1]:
		theta_2_var.append( (second * first) )
# Building the starting posture set
rotation_series = []
for first in theta_0_var:
	for second in theta_1_var:
		for third in theta_2_var:
			rotation_series.append([first, second, third])
#rotation_series = [[quat(0.05, [0,0,1]), quat(0.05, [0,0,1]), quat(0.05, [0,0,1])], 
#	[quat(1, [0,0,1]), quat(1, [0,0,1]), quat(1, [0,0,1])],
#	[quat(1.2, [0,0,1]), quat(1.2, [0,0,1]), quat(1.2, [0,0,1])]]
# Two Target posture definition - fully stretched out and folded together
target_series = [[quat(0.02, [0,0,1]), quat(0.02, [0,0,1]), quat(0.02, [0,0,1])], 
	[quat(1.58, [0,0,1]), quat(1.58, [0,0,1]), quat(1.58, [0,0,1])] ]

#######################################	
# Setup Dual Quaternion MMC  network
#######################################
# Setup visualization class for distance plot in the end. 
distancePlotting = DataVisualizerPlot([1.,1.,1.]) #[1.3963, -0.5236]) 

# Load simplified MMC Arm Network
from dualQuaternionMmcArmNetwork import DualQuaternionMMCArmNetwork
from pycgtypes.quat import slerp as slerp

test = DualQuaternionMMCArmNetwork()
i = 0
it_step = 51
number_targets = len(rotation_series)
data_norm_distance = [ [ [] for i in range(number_targets) ] for j in range(number_targets) ]
target_quat = [0,0,0]

#######################################	
# Run simulations: for each start posture to all target postures
#######################################	
for start_posture in range(0,len(rotation_series)):
	for target_posture in range(0,len(target_series)):
		####
		# Drive arm network into start posture
		for i in range(0,50):
			test.theta_0[0].real_part = quat(rotation_series[start_posture][0])
			test.theta_1[0].real_part = quat(rotation_series[start_posture][1])
			test.theta_2[0].real_part = quat(rotation_series[start_posture][2])
			test.mmc_kinematic_iteration_step()
			#test.r[0].set_translation(vec3([0.05, 0.05, 3.0]))
			#test.alpha[0].set_rotation(0,[0,0,1])
			#test.set_joint_angles(angle_series[start_posture])
# 			target_angles = angle_series[target_posture]
# 			angles = test.get_joint_angles()
# 			diff_angles = (target_angles - angles)/25
# 		# Calculate Normalization factor = distance between start and target point
		#print("Kin: ",test.forwardKinematics(rotation_series[start_posture]))
		target_vect = test.forwardKinematics(target_series[target_posture])
		diff_vect = test.forwardKinematics(rotation_series[start_posture]) - target_vect
 		start_target_dist = numpy.sqrt( diff_vect[0]**2 + diff_vect[1]**2 + diff_vect[2]**2 )		
 		####
 		# Drive from start to target posture for 25 iteration
 		# afterwards settle for 25 more steps
 		for i in range(0,it_step) :
 			if i<26:
 				target_quat[0] = slerp( (i*0.04), rotation_series[start_posture][0], target_series[target_posture][0])
 				target_quat[1] = slerp( (i*0.04), rotation_series[start_posture][1], target_series[target_posture][1])
 				target_quat[2] = slerp( (i*0.04), rotation_series[start_posture][2], target_series[target_posture][2])
			test.theta_0[0].real_part = quat(target_quat[0])
			test.theta_1[0].real_part = quat(target_quat[1])
			test.theta_2[0].real_part = quat(target_quat[2])
			test.mmc_kinematic_iteration_step()
			test.mmc_kinematic_iteration_step()
			#test.mmc_dynamic_iteration_update()
			#print(test.get_target_coordinates() , test.theta_2[0].real_part)
# 				diff_vect = test.get_end_point_in_coxa_cs() - test.forwardKinematik(angle_series[target_posture])
			# Record over time  the distance from end effector to target point
			temp_vect = test.get_target_coordinates()
			target_mmc_vect = numpy.array([temp_vect[0][1], temp_vect[1][1],temp_vect[2][1]])
			#temp_vect = test.get_manipulator_coordinates()
			#posture_vect = numpy.array([temp_vect[0][3], temp_vect[1][3], temp_vect[2][3]])
			diff_vect = target_vect - target_mmc_vect 
 			dist = numpy.sqrt( diff_vect[0]**2 + diff_vect[1]**2 + diff_vect[2]**2 ) / start_target_dist
 			data_norm_distance[start_posture][target_posture].append(dist)
# 				#print(dist, angles, test.get_joint_angles(), test.forwardKinematik(angles), test.forwardKinematik(test.get_joint_angles()))
 		print('### New Movement: ', start_posture, target_posture, dist, start_target_dist)

#######################################	
# Statistics over the movement:
#######################################	
stat_list_of_values_over_time = [ [] for j in range(it_step) ]
stat_mean = [ [] for j in range(it_step) ]
stat_stdev = [ [] for j in range(it_step) ]
for time_steps in range(0,it_step):
	for start_point in range(0,len(rotation_series)) :
		for end_point in range(0,len(target_series)) :
#			if end_point != start_point :
			stat_list_of_values_over_time[time_steps].append(data_norm_distance[start_point][end_point][time_steps])
#     #stat_list_of_values_over_time[time_steps].append(data_norm_distance[1][0][time_steps])
	stat_mean[time_steps] = numpy.mean(stat_list_of_values_over_time[time_steps])
	stat_stdev[time_steps] = numpy.std(stat_list_of_values_over_time[time_steps])
stat_lower_limit = [mv - sd for mv, sd in zip(stat_mean, stat_stdev)] 
stat_upper_limit = [mv + sd for mv, sd in zip(stat_mean, stat_stdev)] 
print(stat_mean)
print(stat_stdev)
distancePlotting.draw_distance_curve(stat_mean, stat_lower_limit, stat_upper_limit)
raw_input("Wait")

###################################
# Results for forward computation:
#
#Mean distance over time: [0.99989544784707618, 0.99412388861275403, 0.98358469937716464, 0.96928923638164544, 0.95183148365107806, 0.93157852260139828, 0.90876563596037485, 0.88355296709843811, 0.85605335120393589, 0.8263631954333801, 0.7945742226331628, 0.76077168817975827, 0.72504124259365899, 0.68747478115330973, 0.64817194539125156, 0.60724176018158882, 0.56480352355887031, 0.52098727070179274, 0.47593411123409746, 0.42979678154350198, 0.38274095640625333, 0.33494843202242897, 0.28662491902469928, 0.23802053851950575, 0.18949216931502436, 0.14153889316106794, 0.10875877782454929, 0.085979038732652432, 0.069585259124905791, 0.057380334033287447, 0.048024304141544717, 0.04067393242025092, 0.034780609936877531, 0.029976178022301382, 0.0260061297826857, 0.022689372251269262, 0.019893450456050038, 0.017519102044223994, 0.015489978989887949, 0.013746418036348776, 0.012241100728863004, 0.010935898452360456, 0.00979974132481015, 0.0088070895907526228, 0.0079368039359496833, 0.0071712884970659158, 0.0064958266293146582, 0.0058980574597495916, 0.0053675582200017917, 0.0048955078846320641, 0.004474414391151765]
#Std Dev over time:	      [0.0021290011834589921, 0.0064581645504222475, 0.01381446004233855, 0.022874832208066787, 0.032729915191740525, 0.042778670716857599, 0.052604050592390386, 0.061906910092266842, 0.070460309185766762, 0.078089995660299755, 0.084664493624475909, 0.090080677624772176, 0.09426086685776372, 0.09714994034606568, 0.098713052214223176, 0.098934005570100056, 0.09781395784698256, 0.095370418139092736, 0.091636615250344916, 0.086661447543494599, 0.080510459741697155, 0.073268806518758847, 0.065048488592339138, 0.056006305910869983, 0.046396233814166868, 0.036706831594550696, 0.030400632313673865, 0.025961316346114104, 0.022591751778114393, 0.019912716743458875, 0.017733087299681913, 0.015941373572104952, 0.014458616453097678, 0.013221564534372116, 0.012177869019459489, 0.011284688234978526, 0.010507830069670845, 0.0098205794528208514, 0.0092031493730683235, 0.0086409143508920355, 0.0081232521118134952, 0.0076427693014870617, 0.0071944271269118107, 0.0067748206223639095, 0.0063816261921531879, 0.0060131996342375569, 0.0056682959829916363, 0.0053458820817881472, 0.0050450168492581476, 0.0047647794009346052, 0.0045042300591011551]
