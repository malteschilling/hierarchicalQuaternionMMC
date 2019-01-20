'''
Version from October 2016

@author: mschilling

Version changes: does not require all the in-between variables.
    Simplified to only diagonals incorporating all subsequent rotations.
'''
import pycgtypes.vec3 as vec3
import math
import matplotlib.pylab as py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dualQuaternion import dualQuaternion
import pycgtypes.quat as quat
import numpy

class DualQuaternionMMCArmNetwork:
    """ Three segmented manipulator with three rotational joints.
        Class for representing a Mean of Multiple Computation Network of an Arm.
        As arguments the segment length can be set-up.
        Afterwards all the necessary variables are set-up as transformations,
        represented as Dual Quaternions.
        The arm is fully stretched in initialisation.
    """
    
    def __init__(self, *args):
        """ Setting up the manipulator arm:
            No arguments are used. 
            All necessary variables for the MMC computation are
            initialised as list of Dual Quaternions representing the arm:
                l_x - Segment translations
                d_x - Diagonal translation
                theta_x - joint rotation
                gamma_x - diagonal rotation
                delta_x - additional rotations for aligning orientations
                r - target rotation and translation
            For each variable exists a list which contains:
                [current value,
                 last value,
                 2 or 3 different values computed from different equations] 
            After setup the arm is fully outstretched.
            
            The graphic output is initialised.
        """
        # Initialisation of segments as translations
        self.l_0 = [dualQuaternion(),0,0,0]
        self.l_1 = [dualQuaternion(),0,0,0]
        self.l_2 = [dualQuaternion(),0,0,0]
        if len(args)==0:
            self.l_0[0].set_translation([1.,0,0])
            self.l_1[0].set_translation([1.,0,0])
            self.l_2[0].set_translation([1.0,0,0])
        # Initialisation of Diagonals and End effector Translations
        self.d_0 = [self.l_0[0] * self.l_1[0],0,0,0] 
        self.d_1 = [self.l_1[0] * self.l_2[0],0,0,0]
        self.r = [self.d_0[0] * self.l_2[0],0,0,0]
        # Initialisation of Angles: all set to zero = fully stretched
        self.theta_0 = [dualQuaternion(),0,0,0,0,0]
        self.theta_1 = [dualQuaternion(),0,0,0,0,0]
        self.theta_2 = [dualQuaternion(),0,0,0,0,0]
        self.delta = [dualQuaternion(),0,0,0]
        # Initialisation of Velocities and Accelerations
        self.theta_0_vel = [dualQuaternion(),0,0,0]
        self.theta_1_vel = [dualQuaternion(),0,0,0]
        self.theta_2_vel = [dualQuaternion(),0,0,0]
        self.theta_0_acc = [dualQuaternion(),0,0,0]
        self.theta_1_acc = [dualQuaternion(),0,0,0]
        self.theta_2_acc = [dualQuaternion(),0,0,0]
        
        # Damping values
        self.mmc_damping = 6
        self.mmc_vel_damping = 0
        self.mmc_acc_damping = 0
        
        # Initialising the drawing window
        #py.ion()
        #py.rcParams['figure.figsize'] = 8, 8
        #self.arm_line, = py.plot(self.get_manipulator_coordinates()[0], self.get_manipulator_coordinates()[1], linewidth=3.0, color='green', marker='o',alpha=0.7, mfc='green')
        #self.target_line, = py.plot(self.get_target_coordinates()[0], self.get_target_coordinates()[1], linewidth=1.0, color='red', marker='x',alpha=0.7, mfc='red')
        #py.xlim(-abs(self.r[0].dual_part)*2-0.5,abs(self.r[0].dual_part)*2+0.5)  
        #py.ylim(-abs(self.r[0].dual_part)*2-0.5,abs(self.r[0].dual_part)*2+0.5)
        #py.draw()
        
        # Writing down the distance to target and the velocity
        self.last_end_point = vec3(self.get_manipulator_coordinates()[0][3], \
                  self.get_manipulator_coordinates()[1][3], \
                  self.get_manipulator_coordinates()[2][3])
        self.data_end_point_velocity = []
        self.data_target_distance = []
        self.sensor_fusion = 0
        
    def init_with_joint_configuration(self, thetas):
        """ Initialise all variables in the network to given joint configurations.
            &thetas - three joint orientations given as quaternions
        """
        # MISSING: theta_1 is not one dimensional anymore
        self.theta_0[0] = thetas[0]
        self.theta_1[0] = thetas[1]
        self.theta_2[0] = thetas[2]
        
         # Initialisation of Diagonals and End effector Translations
        self.d_0[0] = self.theta_0[0] * self.l_0[0] * self.theta_1[0] * self.l_1[0]
        self.d_1[0] = self.theta_1[0] * self.l_1[0] * self.theta_2[0] * self.l_2[0]
        
        target_dq = self.theta_0[0] * self.l_0[0] * self.theta_1[0] * self.l_1[0] * \
                      self.theta_2[0] * self.l_2[0]
        target_dq.compensate_rotation()
        self.set_target_vect( (vec3([2*target_dq.dual_part.x, 2*target_dq.dual_part.y, 2*target_dq.dual_part.z])) )
        self.delta[0] = dualQuaternion(self.r[0].inverse() * self.theta_0[0] * self.l_0[0] * self.d_1[0])
        
    def set_target_vect(self, target_vect):
        start_vect = vec3([1., 0., 0.])
        axis = start_vect.cross(target_vect)
        angle = start_vect.angle(target_vect)
        self.r[0] = dualQuaternion( quat(angle, axis), quat() )
        self.r[0].dual_part = self.r[0].real_part * quat([0., target_vect.length()*0.5, 0., 0.])
    
    def get_manipulator_coordinates(self):
        """ The forward kinematics is explicitly computed and x, y, z for
            base - first joint - second joint - third joint - end effector
            are returned.
        """
        seg1 = self.theta_0[0] * self.l_0[0]
        seg2 = seg1 * self.theta_1[0] * self.l_1[0]
        seg3 = seg2 * self.theta_2[0] * self.l_2[0] 
        seg1.compensate_rotation()
        seg2.compensate_rotation()
        seg3.compensate_rotation()
        return [[0, seg1.dual_part.x*2, seg2.dual_part.x*2, seg3.dual_part.x*2],
                [0, seg1.dual_part.y*2, seg2.dual_part.y*2, seg3.dual_part.y*2],
                [0, seg1.dual_part.z*2, seg2.dual_part.z*2, seg3.dual_part.z*2]]
                
    def forwardKinematics(self, joint_angles):
        """ The forward kinematics is explicitly computed and x, y, z for
            base - first joint - second joint - third joint - end effector
            are returned.
        """
        thetas=[dualQuaternion(), dualQuaternion(), dualQuaternion()]
        thetas[0].real_part = joint_angles[0]
        thetas[1].real_part = joint_angles[1]
        thetas[2].real_part = joint_angles[2]
        seg1 = thetas[0] * self.l_0[0]
        seg2 = seg1 * thetas[1] * self.l_1[0]
        seg3 = seg2 * thetas[2] * self.l_2[0] 
        seg1.compensate_rotation()
        seg2.compensate_rotation()
        seg3.compensate_rotation()
        return numpy.array([seg3.dual_part.x*2, seg3.dual_part.y*2, seg3.dual_part.z*2])
        
    def get_target_coordinates(self):
        """ The coordinates of the target position are computed.
        """
        target = dualQuaternion(self.r[0])
        target.compensate_rotation()
        return [[0, target.dual_part.x*2],
                [0, target.dual_part.y*2],
                [0, target.dual_part.z*2]]
        
    def initialise_drawing_window(self):
        """ Initialising the drawing window.
        Must be colled before the visualisation can be updated
        by calling draw_manipulator"""
        py.ion()
        py.rcParams['figure.figsize'] = 8, 8
        self.arm_line, = py.plot(self.get_manipulator_coordinates()[0], self.get_manipulator_coordinates()[1], linewidth=3.0, color='green', marker='o',alpha=0.7, mfc='green')
        self.target_line, = py.plot(self.get_target_coordinates()[0], self.get_target_coordinates()[1], linewidth=1.0, color='red', marker='x',alpha=0.7, mfc='red')
        py.xlim(-abs(self.r[0].dual_part)*2-0.5,abs(self.r[0].dual_part)*2+0.5)  
        py.ylim(-abs(self.r[0].dual_part)*2-0.5,abs(self.r[0].dual_part)*2+0.5)
        py.draw()
        self.iteration = 0
        
    def initialise_drawing_window_3d(self):
        """ Initialising the drawing window.
        Must be colled before the visualisation can be updated
        by calling draw_manipulator"""
        py.ion()
        py.rcParams['figure.figsize'] = 8, 8
        fig = plt.figure()
        self.ax = Axes3D(fig)
        #theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
        
        self.ax.plot([-3,3], [-3,3], [-3,3], linestyle='None')
        
        self.target_line = self.ax.plot(self.get_target_coordinates()[0], self.get_target_coordinates()[1], self.get_target_coordinates()[2], linewidth=1.0, color='red', marker='x',alpha=0.7, mfc='red')
        
        self.arm_line = self.ax.plot(self.get_manipulator_coordinates()[0], self.get_manipulator_coordinates()[1], self.get_manipulator_coordinates()[2], linewidth=3.0, color='green', marker='o',alpha=0.7, mfc='green')
        
        #self.arm_line = None
        #py.draw()
        #self.target_line = self.ax.plot(self.get_target_coordinates()[0], self.get_target_coordinates()[1], self.get_target_coordinates()[2], linewidth=1.0, color='red', marker='x',alpha=0.7, mfc='red')
        py.xlim(-4,4)
        #py.xlim(-abs(self.r[0])-0.5,abs(self.r[0])+0.5)  
        #py.ylim(-abs(self.r[0])-0.5,abs(self.r[0])+0.5)
        #py.zlim(-abs(self.r[0])-0.5,abs(self.r[0])+0.5)
        py.draw()
        
    def draw_manipulator(self):
        """ The draw method for the manipulator arm.
            It is called from the outside iteration loop.
        """
        self.arm_line.set_xdata(self.get_manipulator_coordinates()[0])
        self.arm_line.set_ydata(self.get_manipulator_coordinates()[1])  # update the data
        self.target_line.set_xdata(self.get_target_coordinates()[0])
        self.target_line.set_ydata(self.get_target_coordinates()[1]) 
        if (self.iteration%5) == 0 :
            fig = plt.figure()        
            plt.plot([-1,3], [-1,3], linestyle='None')
            if self.iteration<10 :
                plt.plot([0,0.5,1.0,1.5], [0,0.71,1.42,2.12], linewidth=5.0, color='0.4', alpha=0.5)
            py.xlim(-3,3)
            py.ylim(-3,3)
            plt.plot(self.get_manipulator_coordinates()[1], self.get_manipulator_coordinates()[2], linewidth=5.0, color='0.25', marker='o',alpha=1.0, mfc='0.75')
            plt.savefig("/Users/mschilling/Desktop/MMC_Constrained_YZ_Iter"+str(self.iteration)+".pdf")
            fig = plt.figure()
        self.iteration += 1
        py.draw()                         # redraw the canvas       
        
    def draw_manipulator_3d(self):
        """ The draw method for the manipulator arm.
            It is called from the outside iteration loop.
        """
        #print self.arm_line
        if self.arm_line is not None:
            self.arm_line[0].set_linestyle('None')
            self.arm_line[0].set_marker('None')
        self.arm_line = self.ax.plot(self.get_manipulator_coordinates()[0], self.get_manipulator_coordinates()[1], self.get_manipulator_coordinates()[2], linewidth=3.0, color='green', marker='o',alpha=0.7, mfc='green')
#        self.arm_line.set_xdata(self.get_manipulator_coordinates()[0])
 #       self.arm_line.set_ydata(self.get_manipulator_coordinates()[1])
        #self.arm_line.set_zdata(self.get_manipulator_coordinates()[2])
        # update the data
        #self.target_line.set_xdata(self.get_target_coordinates()[0])
        #self.target_line.set_ydata(self.get_target_coordinates()[1]) 
        py.draw()                         # redraw the canvas  
        
    def draw_velocity_curve(self, velocity_data=0, mean_distance_data=0, std_dev_low=0, std_dev_up=0 ):
        """
            The velocity profile of the endpoint is drawn.
            Data has to be acquired by calling 
            store_endpoint_data - which
            is automatically called for the dynamic version.
        """
        if velocity_data==0 :
            velocity_data = self.data_end_point_velocity
            norm_distance_data = self.data_target_distance
        if (velocity_data != []) :
            #print "Velocity"
            py.figure(2)
            py.plot(velocity_data)
            py.xlabel('Iteration Steps')
            py.ylabel('Velocity (Units/It.Step)')
            py.title('Velocity Profile of Movement (End point)')
            py.grid(True) 
            py.savefig('velocity_profile.eps')
            py.figure(3)
            py.plot(mean_distance_data, color='0.1')
            py.hold(True)
            py.plot(std_dev_low,'--',color='0.5')
            py.plot(std_dev_up,'--', color='0.5')
            py.xlabel('Iteration Steps')
            py.ylabel('Distance (normalized)')
            py.title('Distance to Target (End point)')
            py.grid(True) 
            py.savefig('distance.eps')
    
    def compute_multiple_computations(self):
        """ For each variable new values are calculated through
            different equations. The values are pushed on the variable
            stack
        """
        # Theta 0:
        self.theta_0[2] = self.d_0[0] * self.l_1[0].inverse() * self.theta_1[0].inverse() * self.l_0[0].inverse()
        self.theta_0[3] = self.r[0] * self.delta[0] * self.d_1[0].inverse() * self.l_0[0].inverse()
        #print(self.delta[0])
        
        # Theta 1:
        self.theta_1[2] = self.l_0[0].inverse() * self.theta_0[0].inverse() * self.d_0[0] * self.l_1[0].inverse() 
        self.theta_1[3] = self.d_1[0] * self.l_2[0].inverse() * self.theta_2[0].inverse() * self.l_1[0].inverse()
        # Theta 2:
        self.theta_2[2] = self.l_1[0].inverse() * self.theta_1[0].inverse() * self.d_1[0] * self.l_2[0].inverse()  
        self.theta_2[3] = self.d_0[0].inverse() * self.r[0] * self.delta[0] * self.l_2[0].inverse()
            
        # Delta:
        self.delta[2] = self.r[0].inverse() * self.theta_0[0] * self.l_0[0] * self.d_1[0]
        self.delta[3] = self.r[0].inverse() * self.d_0[0] * self.theta_2[0] * self.l_2[0] 
            
        # D_0:
        self.d_0[2] = self.theta_0[0] * self.l_0[0] * self.theta_1[0] * self.l_1[0]
        self.d_0[3] = self.r[0] * self.delta[0] * self.l_2[0].inverse() * self.theta_2[0].inverse()
        # D_1:
        self.d_1[2] = self.theta_1[0] * self.l_1[0] * self.theta_2[0] * self.l_2[0]
        self.d_1[3] = self.l_0[0].inverse()  * self.theta_0[0].inverse() * self.r[0] * self.delta[0]
        # R:
        self.r[2] = self.theta_0[0] * self.l_0[0] * self.d_1[0] * self.delta[0].inverse()
        self.r[3] = self.d_0[0] * self.theta_2[0] * self.l_2[0] * self.delta[0].inverse()

    def compensate_errors_in_multiple_computations(self):
        """ Align resulting dual quaternions:
            in rotational joints the translational share is compensated and
            in translational transformations the rotation part is compensated.
        """
        self.theta_0[2].compensate_translation(self.l_0[0],1)
        self.theta_0[3].compensate_translation(self.l_0[0],1)
        self.theta_1[2].compensate_translation(self.l_1[0],1)
        self.theta_1[3].compensate_translation(self.l_1[0],1)
        self.theta_2[2].compensate_translation(self.l_2[0],1)
        self.theta_2[3].compensate_translation(self.l_2[0],1)
        # TODO: Compensate in Delta?
        self.delta[2].strip_translation()
        self.delta[3].strip_translation()
        
    def add_new_sensor_readings(self, sensor_readings):
        """ Sensory feedback can be infused into the body model.
        
            Multiple sensor values can be given into the system
            which will be weighted equally as a kinematic equation.
            
            The inputs are given as a list: for each joint a maximum of two sensor
            readings can be provided.
        """        
        self.theta_0[4] = dualQuaternion(sensor_readings[0][0])
        self.theta_0[5] = dualQuaternion(sensor_readings[0][1])
        self.theta_1[4] = dualQuaternion(sensor_readings[1][0])
        self.theta_1[5] = dualQuaternion(sensor_readings[1][1])
        self.theta_2[4] = dualQuaternion(sensor_readings[2][0])
        self.theta_2[5] = dualQuaternion(sensor_readings[2][1])
        self.sensor_fusion = len(sensor_readings[0])
        
    def calculate_mean_values_of_multiple_computations(self, damp):
        """ Calculate the mean value for all the variables.
            This should only be done, after the multiple
            computations have been computed and saved in the
            variable list (starting from third element).
        """
        if (self.sensor_fusion > 0):
            self.theta_0[0].linear_blending(self.theta_0[1:6],[damp,1,1,1,1])
            self.theta_1[0].linear_blending(self.theta_1[1:6],[damp,1,1,1,1])
            self.theta_2[0].linear_blending(self.theta_2[1:6],[damp,1,1,1,1])
        else:
            self.theta_0[0].linear_blending(self.theta_0[1:4],[damp,1,1])
            self.theta_1[0].linear_blending(self.theta_1[1:4],[damp,1,1])
            self.theta_2[0].linear_blending(self.theta_2[1:4],[damp,1,1])
        self.delta[0].linear_blending(self.delta[1:4],[damp/4,1,1])
        self.r[0].linear_blending(self.r[1:4],[damp,1,1])
        self.d_0[0].linear_blending(self.d_0[1:4],[damp/4,1,1])
        self.d_1[0].linear_blending(self.d_1[1:4],[damp/4,1,1])
            
    def mmc_kinematic_iteration_step(self):
        """ The MMC Method: 
            - the multiple computations are computed for each variable
            - afterwards: a normalisation is done
                for joints: translational share is compensated
                for translations: rotational share is compensated
            - the mean for each variable is calculated
            Before this method is called the input values has to be enforced on 
            the variables every time: 
                e.g., for inverse kinematic = r[0] and alpha[0] has to be set
            The old value is shifted to the second element in the lists.                
        """        
        # Remember the old values of the variables
        self.theta_0[1]=dualQuaternion(self.theta_0[0])
        self.theta_1[1]=dualQuaternion(self.theta_1[0])
        self.theta_2[1]=dualQuaternion(self.theta_2[0])
        self.delta[1]=dualQuaternion(self.delta[0])
        self.d_0[1]=dualQuaternion(self.d_0[0])
        self.d_1[1]=dualQuaternion(self.d_1[0])
        self.r[1]=dualQuaternion(self.r[0])
        
        # Apply multiple computations:
        # First, the geometric relationships are used
        # - i.e. the equations describing the triangles.
        # Second, for the second and third joint the cosine rule
        # is in addition used. This is necessary to flex the movement
        # when the target is very nearby (the cosine rule computes the enclosed
        # angle between the two joining segments depending on the length of the
        # opposing diagonal).
        self.compute_multiple_computations()
        
        # Align resulting dual quaternions:
        # in rotational joints the translational share is compensated and
        # in translational transformations the rotation part is compensated
        self.compensate_errors_in_multiple_computations()
        
        # Interpolate the multiple computations for each variable.
        # In addition, the current value weighted by the parameter
        # mmc_damping is also included in the calculation.
        # This prevents oscillations as the current value is fed back into
        # the system.
        self.calculate_mean_values_of_multiple_computations(self.mmc_damping)
        
        # Constraint application: Elbow joint is restricted to one degree of freedom
        # TODO: CHANGE BACK TO ONE DIMENSIONAL
        self.theta_1[0].project_onto_fixed_rotation(vec3(0,0,1))

    def calculate_dynamic_values(self):
        """ 
            From the current joint angle values
            and the old joint angle values the current velocities 
            and the accelerations are computed.
            Before this, the new joint values must have been set
            (mmcKinematicIterationStep)
        """
        # Remember the old values of the variables
        self.theta_0_vel[1]=dualQuaternion(self.theta_0_vel[0])
        self.theta_1_vel[1]=dualQuaternion(self.theta_1_vel[0])
        self.theta_2_vel[1]=dualQuaternion(self.theta_2_vel[0])
        self.theta_0_acc[1]=dualQuaternion(self.theta_0_acc[0])
        self.theta_1_acc[1]=dualQuaternion(self.theta_1_acc[0])
        self.theta_2_acc[1]=dualQuaternion(self.theta_2_acc[0])
        
        self.theta_0_vel[0].linear_blending([self.theta_0_vel[1], \
                                            (self.theta_0[1].get_conjugate() * self.theta_0[0])], \
                                            [self.mmc_vel_damping,1])
        #print self.theta_0[0]
        #self.theta_1_vel[0].linearBlending([self.theta_1_vel[1], \
         #                                   (self.theta_1[1].getConjugate() * self.theta_1[0])], \
          #                                  [self.mmc_vel_damping,1])
        #self.theta_2_vel[0].linearBlending([self.theta_2_vel[1], \
           #                                 (self.theta_2[1].getConjugate() * self.theta_2[0])], \
            #                                [self.mmc_vel_damping,1])
        
    def update_joints_by_including_dynamic_influences(self):
        """ For the dynamic MMC network including velocities.
            the joint positions depend on the velocity of the joints
            (which internally is calculated using the traditional
             MMC network approach and integrates additional influences)
            Before the network can be updated the
              calculate_dynamic_values function must be called.
            Usually, one simply calls mmc_dynamic_iteration_update
            after the classic MMC iteration step.
            The mmc_dynamic_iteration_update is doing all the steps:
                - calculate velocities
                - update positions
                - store the end position
        """
        #print "Update dynamic"
        #print self.theta_0[0]
        self.theta_0[0]=self.theta_0[1] * self.theta_0_vel[0]
        #print self.theta_0[0]
        self.theta_1[0]=self.theta_1[1] * self.theta_1_vel[0]
        self.theta_2[0]=self.theta_2[1] * self.theta_2_vel[0]
        
    def store_endpoint_data(self,target_point):
        """ Storing the endpoint positions for
            displaying afterwards the velocity profile.
        """
        current_end_point = vec3(self.get_manipulator_coordinates()[0][3], \
                                 self.get_manipulator_coordinates()[1][3], \
                                 self.get_manipulator_coordinates()[2][3])
        self.data_end_point_velocity.append(abs(current_end_point - self.last_end_point))
        self.data_target_distance.append(abs(current_end_point - target_point ))
        #print abs(current_end_point - target_point )
        #print current_end_point
        #print target_point
        self.last_end_point = current_end_point
        
    def mmc_dynamic_iteration_update(self):
        """ The dynamic iteration step function -
            which is called after the classic MMC iteration step.
            The mmc_dynamic_iteration_update is doing all the steps:
                - calculate velocities
                - update positions
                - store the end position
        """
        self.calculate_dynamic_values()
        self.update_joints_by_including_dynamic_influences()
        self.store_endpoint_data()