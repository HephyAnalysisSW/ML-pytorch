''' Matrix representation of the U1 group 
'''

import numpy as np

class U1:

    def __init__( self, N_channels ):

        self.N_channels = N_channels

        # unit element
        self.unity      = np.diag(np.ones(N_channels)).astype('int') 

        # one step right
        step_right = np.eye(self.N_channels,k=-1)+np.eye(self.N_channels,k=-1+self.N_channels).astype('int')

        # list of group element matrix reps
        self.elements = np.array( [ np.linalg.matrix_power( step_right, power ) for power in np.arange( N_channels ) ] ).astype(int)

        self.N_elements = len(self.elements)

if __name__=="__main__":
    G = U1(10) 
