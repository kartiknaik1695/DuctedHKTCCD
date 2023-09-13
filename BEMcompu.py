import openmdao.api as om
import numpy as np
from math import pi
import pickle
from math import log
import matplotlib.pyplot as plt
import pickle
from scipy.interpolate import interp2d


class Computation(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int,
                             desc='Number of nodes to be evaluated in the RHS')

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('x', val=np.zeros(nn), desc='velocity', units='rad/s')
        self.add_input('xdata', val=np.zeros(6))
        
        self.add_input('time_phase', units='s', val=np.ones(nn))
        self.add_output('torque', val=np.ones(nn))


        r = np.arange(nn)

        self.declare_partials('torque', ['x', 'time_phase'], rows=r, cols=r, method = 'fd')
        self.declare_partials("torque", "xdata",method = 'fd')
        


    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        
        nn = self.options['num_nodes']
        

        xdata = inputs['xdata']
        

        time = inputs['time_phase']
        A = np.pi*1.4**2
        rho = 1000

        # Inflow: sinusoidal
        Uinf = np.sin(0.25 * time) * 0.2 + 1.5
        
        Omega = inputs['x']   # in rad/s        
        Rtip = 0.511*1.4/0.7593546006
        TSR = Omega*Rtip/Uinf
        
        CpVec = np.zeros(nn)
        
        filename = "MRF.pkl"
       
        with open(filename, "rb") as f:
            sm = pickle.load(f)

        for i in range(len(TSR)):
            x_data = np.array([xdata[0], xdata[1], xdata[2], xdata[3], xdata[4], xdata[5], TSR[i]])
            x_data = x_data.reshape(1,-1)
            CpVec[i] = sm.predict_values(x_data)[0][1]

        Tau = 0.5*rho*A*(CpVec/Omega)*np.power(Uinf,3)        
        
        outputs['torque'] = Tau

    # def compute_partials(self, inputs, partials):
        
    #     nn = self.options['num_nodes']
        

    #     time = inputs['time_phase']
        

    #     Uinf = np.sin(0.25 * time) * 0.2 + 1.5
        
    #     Omega = inputs['x'] # convert to RPM
        

    #     ##
    #     dUinf_dtime = np.cos(0.25 * time) * 0.05
        

    #     for i in range(nn):
    #         partials['torque', 'x'][i] = np.squeeze(dQ['dOmega'])[i][i] * 30.0 / pi
    #         partials['torque', 'time_phase'][i] = np.squeeze(dQ['dUinf'])[i][i] * dUinf_dtime[i]








