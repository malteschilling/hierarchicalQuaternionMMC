'''
Created on 9.03.2010

@author: mschilling
'''
import pycgtypes.quat as quat
import pycgtypes.vec3 as vec3
import types, math

class dualQuaternion:
    """Dual Quaternion class
        two quaternions are used to represent transformations
        in general
            - real part: represents a rotation
            - dual part: represents a translation
        
        As a geometric interpretation, a dual quaternion describes
        a screw motion - i.e. a rotation around an axis and a 
        translation along this axis. It is defined by the angle of
        rotation, the length of translation, the axis of rotation 
        (and direction of translation) as well as the offset of this
        center of rotation (the axis).
        
        Transformations are concatenated by right multiplication 
        (with respect to the new transformed coordinate frame). 
        
        By convention the transformations are provided in a half
        notation = half angle of rotation, half translation because
        when actually used for calculating positions, one 
        is applying the dual quaternion from both sides:
            q * v * q^-1
    
    """
    
    def __init__(self, *args):
        """
            A dual quaternion can be initialized by giving it a
                - dual quaternion (makes a copy of real and dual part)
                - two quaternions (real, dual part)
                - two lists with four components (one for each quaternion,
                  first = real, second = dual part) 
                - no argument: a null transformation is initialized
        """
        if len(args)==0 :
            self.real_part = quat([1,0,0,0])
            self.dual_part = quat([0,0,0,0])
        elif len(args)==1:
            # One DualQuaternion is given
            if isinstance(args[0], dualQuaternion):
                self.real_part = quat(args[0].real_part)
                self.dual_part = quat(args[0].dual_part)
            # All the values are given in one list
            else:
                self.real_part = (quat((args[0])[0:4])).normalize()
                self.dual_part = quat((args[0])[4:])
        elif len(args)==2:
            # Two quaternions are given
            if isinstance(args[0], quat):
                self.real_part = quat(args[0])
                self.dual_part = quat(args[1])
            # Two lists are given
            else :
                self.real_part = (quat(args[0])).normalize()
                self.dual_part = quat(args[1])
        
    def set_rotation (self, rot_angle, rot_axis):
        """Initialize by providing an Angle and an axis - as a vector (.,.,.)"""
        self.real_part = quat(rot_angle, rot_axis)
        
    def set_translation (self, transl_list):
        """ Initialize by providing a Translation as a list"""
        self.dual_part = quat([0] + [x*0.5 for x in transl_list])   
    
    def __mul__ (self,post_op):
        """ Multiplication - of two Dual quaternions or scaling.
            The dual quaternion is (post-)multiplied with 
                - a scalar (float, int or long)
                - another dual quaternion
        """
        T = type(post_op)
        # Dual Quaternion * scalar
        if T==float or T==int or T==int:
            return dualQuaternion(self.real_part*post_op, self.dual_part*post_op)
        # DualQuaternion*DualQuaternion
        if isinstance(post_op, dualQuaternion):
            return dualQuaternion(self.real_part * post_op.real_part,
                                  self.real_part * post_op.dual_part +
                                  self.dual_part * post_op.real_part)
        else:
            raise TypeError("unsupported operand type for *")
        
    def __str__(self):
        """ Output method for dual quaternion:
            first real and then dual part is given back.
        """
        return '('+self.real_part.__str__()+' '+self.dual_part.__str__()+')'
        
    def get_conjugate(self):
        """ Conjugation is working on each of the quaternions individually
            but for the translational part one also has to invoke the
            conjugate of the dual - therefore the -1"""
        return dualQuaternion(self.real_part.conjugate(),self.dual_part.conjugate()*-1)

    def inverse(self):
        """ Inverse is computed for the real part as for the quaternions 
            (means it is switching signs and scaling by the lenght which
             has no effect as the rotations are unit length)
            Again, for the dual part this is different - the inverse is switching
            signs but no scaling. From the geometric point of view this becomes
            obvious as there is no reason to normalize an inverse translation to unit
            length. Therefore, we use simply conjugate here (the mathematical
            explanation involves the definition of the dual quaternion multiplication)"""
        return dualQuaternion(self.real_part.inverse(),self.dual_part.conjugate())
    
    def normalize(self):
        """" Normalize the dual quaternion - afterwards the real part
            is unit length.
        """
        w_dq = abs(self.real_part)
        self.real_part = self.real_part * 1/w_dq
        # ??? Scale also dual part?
        self.dual_part = self.dual_part * 1/w_dq 
    
    def strip_translation(self):
        """ For joint transformations: only rotations should be left -
            the translational part should be cancelled (or compensated)"""
        self.dual_part = quat([0,0,0,0])
        
    def strip_rotation(self):
        """ For segment transformations: only translations should be left -
            the rotational part should be cancelled (or compensated)"""
        self.real_part = quat([1,0,0,0])
        
    def compensate_rotation(self):
        """ For segment transformations: only translations should be left -
            the rotational part (real part) can be compensated.
            Here, the complete displacement is calculated - what is lost
            is the orientation of the endpoint """
        self.dual_part = (self * self.get_conjugate()).dual_part * 0.5
        self.real_part = quat([1,0,0,0])
        
    def project_onto_fixed_rotation(self, axis_vec3):
        """ Takes the current rotation angle and uses it with the fixed rotation axis.
            This is not veyr accurate, but the error is minimized in the right direction
            and the network converges over time.
            
            Important: has to check if the different axis are not pointing into opposite
            directions (angle between is smaller 180 degrees) """
        angle_cur, axis_cur = self.real_part.toAngleAxis()
        if (abs(axis_cur) > 0):
            if (axis_cur.angle(axis_vec3) > math.pi/2):
                angle_cur = -angle_cur
    	self.real_part = quat(angle_cur, axis_vec3)
        
    def compensate_translation(self, along_seg, scale=1):
        """ For joint transformations: only rotations should be left -
            the translational part (dual part) can be partly compensated.
            The part of the displacement which is orthogonal to the outgoing
            segment can be compensated through rotation of the joint. 
            The part along the following segment must be simply dropped """
        # Here the translational error is calculated.
        # ! With respect to the rotated reference frame!
        # Therefore here the order has to be quat.conjugate * quat
        # - not the other way around
        error_dq = self.get_conjugate() * self
        # Because the multiplication is doubling the length,
        # it has to be multiplied by 0.5 and can be scaled
        error_dq.dual_part = error_dq.dual_part * scale * 0.5
        # The two translational parts of the dual quaternions are
        # transformed to vectors - for using dot product 
        error_vec = vec3(error_dq.dual_part.x,error_dq.dual_part.y,error_dq.dual_part.z)
        along_vec = vec3(along_seg.dual_part.x,along_seg.dual_part.y,along_seg.dual_part.z)
        dist = along_seg.dual_part.dot(error_dq.dual_part)/abs(along_seg.dual_part)
        # The subtracted term is the projection of the error term onto 
        # the other vector (the translation along the segment)
        # The result is therefore the share of the error which is 
        # orthogonal.
        error_dq.dual_part = error_dq.dual_part - dist * along_seg.dual_part.normalize() 
        # Only the share of the error which is orthogonal to the 
        # outgoing segment shall be used.
        # First: the translational share is dropped.
        self.strip_translation()
        # Then the current rotation is furthermore rotated by a rotation which
        # would align the current outgoing segment with the direction of 
        # the intended orientation (given through the concatenation of the 
        # segment and the orthogonal error).
        self.real_part=self.real_part*(along_seg.get_aligning_rotation(along_seg*error_dq))
        
    def get_aligning_rotation(self, target):
        """ Returns a rotation (a quaternion) which rotates self (translational
            part) onto the target dual quaternion (translational part) """
        start_vec = vec3(self.dual_part.x,self.dual_part.y,self.dual_part.z)
        target_vec = vec3(target.dual_part.x,target.dual_part.y,target.dual_part.z)
        try:
            comp_angle = start_vec.angle(target_vec)
        except ValueError:
            comp_angle = 0.0
            #print "ValueError caught"
        comp_axis = start_vec.cross(target_vec)
        return quat([comp_angle,comp_axis])   
    
    def linear_blending(self,quat_list,weights):
        """ Linear interpolation between a list of dual quaternions.
            In addition, a list with weights has to be provided:
             each dual quaternion is weighted by the corresponding factor
             of this second list.
             Linear interpolation simply calculates a mean dual quaternion
             and then normalizes (with respect to the real part) it.
        """ 
        sum_weights=sum(weights)
        result_quat_real = quat([0,0,0,0])
        result_quat_dual = quat([0,0,0,0])
        # Add up the dual quaternions.
        for i in range(len(quat_list)):
            result_quat_real = result_quat_real + \
                quat_list[i].real_part * weights[i]/sum_weights 
            result_quat_dual = result_quat_dual + \
                quat_list[i].dual_part * weights[i]/sum_weights
        self.real_part = result_quat_real
        self.dual_part = result_quat_dual
        self.normalize() 