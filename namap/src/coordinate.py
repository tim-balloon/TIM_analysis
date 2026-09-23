import numpy as np

class rotate():
    '''
    Pitch is a rotation around the x axis
    Yaw is a rotation around the z axis
    Roll is a rotation around the y axis
    To go from inertial to gondola rotate zxy (first yaw, then pitch and then roll)

    Parameters
    ----------

    Returns
    -------
    '''

    def __init__(self, yaw, pitch, roll):

        '''
        Create an instance of the class rotate
        
        Parameters
        ----------
        yaw : float or array-like
            Initial yaw angle(s), in radians.
        pitch : float or array-like
            Initial pitch angle(s), in radians.
        roll : float or array-like
            Initial roll angle(s), in radians.

        Returns
        -------
        '''

        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll

    def rotmatrix(self, yaw_mat = None, pitch_mat=None, roll_mat = None):
        '''
        Compute the rotation matrices for yaw, pitch, and roll.

        Parameters
        ----------
        yaw_mat : float, optional
            Yaw angle used to construct the rotation matrix, in radians
            If not provided, the object's yaw angle is used
        pitch_mat : float, optional
            Pitch angle used to construct the rotation matrix, in radians
            If not provided, the object's pitch angle is used
        roll_mat : float, optional
            Roll angle used to construct the rotation matrix, in radians
            If not provided, the object's roll angle is used

        Returns
        -------
        pitchMatrix : numpy.matrix
            Rotation matrix corresponding to the pitch angle
        rollMatrix : numpy.matrix
            Rotation matrix corresponding to the roll angle
        yawMatrix : numpy.matrix
            Rotation matrix corresponding to the yaw angle
        '''
                
        if yaw_mat is None:
            yaw_mat = self.yaw.copy()
            roll_mat = self.roll.copy()
            pitch_mat = self.pitch.copy()
        yawMatrix = np.matrix([[np.cos(yaw_mat), -np.sin(yaw_mat), 0], \
                               [np.sin(yaw_mat), np.cos(yaw_mat), 0], \
                               [0, 0, 1]])

        rollMatrix = np.matrix([[np.cos(roll_mat), 0, -np.sin(roll_mat)],\
                                [0, 1, 0],\
                                [np.sin(roll_mat), 0, np.cos(roll_mat)]])

        pitchMatrix = np.matrix([[1, 0, 0],\
                                 [0, np.cos(pitch_mat), -np.sin(pitch_mat)],\
                                 [0, np.sin(pitch_mat), np.cos(pitch_mat)]]) 

        return pitchMatrix, rollMatrix, yawMatrix

    def offset_mat(self, yaw_off, pitch_off, roll_off, rot_mat=np.diag(np.ones(3))):
        '''
        Compute a rotation matrix including yaw, pitch, and roll offsets

        Parameters
        ----------
        yaw_off : float
            Yaw offset angle, in radians
        pitch_off : float
            Pitch offset angle, in radians
        roll_off : float
            Roll offset angle, in radians
        rot_mat : numpy.ndarray, optional
            Initial rotation matrix. Defaults to the identity matrix

        Returns
        -------
        rot3: numpy.ndarray
            Rotation matrix after applying the specified angular offset
        '''
                
        pitch_off_mat = self.rotmatrix(yaw_mat = yaw_off, roll_mat = roll_off, pitch = pitch_off)[0]
        roll_off_mat = self.rotmatrix(yaw_mat = yaw_off, roll_mat = roll_off, pitch = pitch_off)[1]
        yaw_off_mat = self.rotmatrix(yaw_mat = yaw_off, roll_mat = roll_off, pitch = pitch_off)[2]

        rot1 = np.matmul(yaw_off_mat, rot_mat)
        rot2 = np.matmul(pitch_off_mat, rot1)
        rot3 = np.matmul(roll_off_mat, rot2)

        return rot3
    
    def offset_angle(self, yaw_off=0., pitch_off=0., roll_off=0., rot_mat=0.):
        '''
        Compute the final angular offsets from a rotation matrix.

        Parameters
        ----------
        yaw_off : float or array-like, optional
            Yaw offset angle(s), in radians
        pitch_off : float or array-like, optional
            Pitch offset angle(s), in radians
        roll_off : float or array-like, optional
            Roll offset angle(s), in radians
        rot_mat : numpy.ndarray or scalar, optional
            Initial or final rotation matrix. If a rotation matrix is
            provided, it is used directly

        Returns
        -------
        pitch_off_final : float
            Final pitch offset angle, in radians
        roll_off_final : float
            Final roll offset angle, in radians
        yaw_off_final : float
            Final yaw offset angle, in radians
        '''
        if np.size(yaw_off) == 1:
            if np.greater(yaw_off,0.) is True or np.greater(roll_off,0.) is True or \
               np.greater(pitch_off,0.) is True:

                rot_matrix = self.offset_mat(yaw_off, pitch_off, roll_off)
        else:
            if np.any(np.greater(yaw_off,0.)) is True or np.any(np.greater(pitch_off,0.)) is True or \
               np.any(np.greater(roll_off,0.)) is True:

                matrix = np.diag(np.ones(3))
                for i in len(yaw_off):
                    rot_matrix = self.offset_mat(yaw_off[i], pitch_off[i], roll_off[i], rot_mat=matrix)
                    matrix = rot_matrix.copy()

        if np.size(rot_mat) >= 3:

            rot_matrix = rot_mat

        pitch_off_final = np.arctan2(rot_matrix[1,2],np.sqrt(rot_matrix[1,0]**2+rot_matrix[1,1]**2))
        roll_off_final = np.arctan2(rot_matrix[0,2],rot_matrix[2,2])
        yaw_off_final = np.arctan2(rot_matrix[1,0],rot_matrix[1,1])

        return pitch_off_final, roll_off_final, yaw_off_final

    def finalcoord(self, yaw_off=0., pitch_off=0., roll_off=0.):
        '''
        Compute the final yaw, pitch, and roll coordinates after applying offsets.

        Parameters
        ----------
        yaw_off : float, optional
            Yaw offset angle, in radians
        pitch_off : float, optional
            Pitch offset angle, in radians     
        roll_off : float, optional
            Roll offset angle, in radians 

        Returns
        -------
        pitch_final : float or array-like
            Final pitch angle(s), in radians
        roll_final : float or array-like
            Final roll angle(s), in radians
        yaw_final : float or array-like
            Final yaw angle(s), in radians
        '''
        cr = np.cos(self.roll)
        sr = np.sin(self.roll)
        cp = np.cos(self.pitch)

        yaw_final = self.yaw+2*np.arcsin(np.sin((yaw_off*cr+pitch_off*sr)/2.)/cp)
        roll_final = self.roll
        pitch_final = self.pitch+(-yaw_off*sr+pitch_off*cr)

        return pitch_final, roll_final, yaw_final