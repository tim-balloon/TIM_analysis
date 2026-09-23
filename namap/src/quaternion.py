import numpy as np

class quaternions(object):
    '''
    Class for quaternion operations and conversions between Euler angles
    and quaternions.

    The Euler angles are defined using the following rotation axes:
    - pitch: rotation about the x-axis
    - roll: rotation about the y-axis
    - yaw: rotation about the z-axis

    The rotations are applied in the z-x-y order, corresponding to
    yaw, followed by pitch, followed by roll.

    Notes
    -----
    Euler angles are expressed in degrees for the input and output of
    the conversion methods. Quaternions are represented as arrays in
    the order [w, x, y, z].
            
    Parameters
    ----------

    Returns
    -------
    '''


    def eul2quat(self, yaw, pitch, roll):

        '''
        Convert Euler angles to a quaternion.

        The quaternion is computed according to the rotation convention
        defined for this class, such that the resulting quaternion is
        equivalent to::

            Q = Q_yaw * Q_pitch * Q_roll

        Parameters
        ----------
        yaw : float
            Yaw angle in degrees. This corresponds to a rotation about the z-axis.
        pitch : float
            Pitch angle in degrees. This corresponds to a rotation about the x-axis.
        roll : float
            Roll angle in degrees. This corresponds to a rotation about the y-axis.

        Returns
        -------
        quaternion_list : list
            Quaternion represented as [w, x, y, z]
        '''

        yaw = np.radians(yaw)
        roll = np.radians(roll)
        pitch = np.radians(pitch)

        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qx = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qy = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)

        quaternion_list = np.array([qw, qx, qy, qz])

        return quaternion_list

    def quat2eul(self, q):

        """
        Convert a quaternion to Euler angles.

        Parameters
        ----------
        q : numpy.ndarray
            Quaternion represented as [w, x, y, z].

        Returns
        -------
        yaw : float
            Yaw angle in degrees, corresponding to a rotation about the z-axis.
        pitch : float
            Pitch angle in degrees, corresponding to a rotation about the x-axis.
        roll : float
            Roll angle in degrees, corresponding to a rotation about the y-axis.
        """

        pitch = np.arcsin(2*q[0]*q[1]-2*q[2]*q[3])

        roll = np.arctan2(2*q[0]*q[2]+2*q[1]*q[3], (1 - 2*(q[1]**2 + q[2]**2)) )

        yaw = np.arctan2(2*q[0]*q[3]+2*q[1]*q[2], (1 - 2*(q[1]**2 + q[3]**2)) )

        return np.degrees(yaw), np.degrees(pitch), np.degrees(roll)

    def product(self, q1, q2):

        """
        Compute the product of two quaternions.

        The order of the input quaternions matters because quaternion
        multiplication is not commutative.

        Parameters
        ----------
        q1 : numpy.ndarray
            First quaternion, represented as [w, x, y, z].
        q2 : numpy.ndarray
            Second quaternion, represented as [w, x, y, z].

        Returns
        -------
        qfinal : numpy.ndarray
            Quaternion corresponding to the product q1 * q2,
            represented as [w, x, y, z].
        """
        

        w0, x0, y0, z0 = q1
        w1, x1, y1, z1 = q2
        qfinal = np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                         x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                         -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                         x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)
        return qfinal

    def reciprocal(self, q):
        """
        Compute the reciprocal (inverse) of a quaternion.

        Parameters
        ----------
        q : numpy.ndarray
            Quaternion represented as [w, x, y, z].

        Returns
        -------
        qm1 : numpy.ndarray
            Reciprocal of the input quaternion, represented as [w, x, y, z].
        """
        

        qm1 = q*np.array([1,-1,-1,-1])/np.sum(q**2)
        return qm1
