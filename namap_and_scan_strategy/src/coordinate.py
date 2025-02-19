import numpy as np

class rotate():

    '''
    Pitch is a rotation around the x axis
    Yaw is a rotation around the z axis
    Roll is a rotation around the y axis
    To go from inertial to gondola rotate zxy (first yaw, then pitch and then roll)
    '''

    def __init__(self, yaw, pitch, roll):

        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll

    def rotmatrix(self, yaw_mat = None, roll_mat = None, pitch_mat=None):
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

        pitch_off_mat = self.rotmatrix(yaw_mat = yaw_off, roll_mat = roll_off, pitch = pitch_off)[0]
        roll_off_mat = self.rotmatrix(yaw_mat = yaw_off, roll_mat = roll_off, pitch = pitch_off)[1]
        yaw_off_mat = self.rotmatrix(yaw_mat = yaw_off, roll_mat = roll_off, pitch = pitch_off)[2]

        rot1 = np.matmul(yaw_off_mat, rot_mat)
        rot2 = np.matmul(pitch_off_mat, rot1)
        rot3 = np.matmul(roll_off_mat, rot2)

        return rot3
    
    def offset_angle(self, yaw_off=0., pitch_off=0., roll_off=0., rot_mat=0.):

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

        cr = np.cos(self.roll)
        sr = np.sin(self.roll)
        cp = np.cos(self.pitch)

        yaw_final = self.yaw+2*np.arcsin(np.sin((yaw_off*cr+pitch_off*sr)/2.)/cp)
        roll_final = self.roll
        pitch_final = self.pitch+(-yaw_off*sr+pitch_off*cr)

        return pitch_final, roll_final, yaw_final