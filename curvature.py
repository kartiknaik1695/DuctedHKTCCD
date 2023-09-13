import openmdao.api as om
import numpy as np
import copy

class curvature(om.ExplicitComponent):

    def setup(self):

        self.add_input("outlet_radius1_scale", val=0.94347634, units=None)
        self.add_input("outlet_radius2_scale", val=0.96796378, units=None)
        self.add_input("outlet_radius3_scale", val=0.95262492, units=None)
        self.add_input("outlet_length_scale", val=1.0, units=None)
        self.add_input("inlet_length_scale", val=1.0, units=None)

        self.add_output("curvature", val=np.zeros(5))

        self.declare_partials("curvature","outlet_radius1_scale", method="fd")
        self.declare_partials("curvature","outlet_radius2_scale", method="fd")
        self.declare_partials("curvature","outlet_radius3_scale", method="fd")
        self.declare_partials("curvature","outlet_length_scale", method="fd")
        self.declare_partials("curvature","inlet_length_scale", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_ouputs=None):

        optimized_radius = np.array([0.7593546006, 0.7164330961, 0.6934812846, 0.6606275535, 0.6126413944, 0.574647822, 0.5656752046, 0.626536342])
        optimized_section = np.array([-0.6297133565, -0.6044429332, -0.5770231068, -0.5244429332, -0.4170231068, -0.2044429332, 0, 0.2202349105])

        initial_scale = optimized_radius[:]/optimized_radius[0]
        initial_scale[1:4] = optimized_radius[1:4]/optimized_radius[0:3]

        # print("initial scale: ", initial_scale)

        outlet_radius_scale = initial_scale[:]

        outlet_length_scale = inputs["outlet_length_scale"]
        inlet_length_scale = inputs["inlet_length_scale"]

        outlet_radius_scale_dv = np.array([inputs["outlet_radius1_scale"][0], inputs["outlet_radius2_scale"][0], inputs["outlet_radius3_scale"][0]])

        # print("outlet_radius_scale_dv: ", outlet_radius_scale_dv)

        outlet_radius_scale[1:4] = outlet_radius_scale_dv[:]

        new_section = np.zeros_like(optimized_section)
        new_section[0:6] = optimized_section[0:6]*outlet_length_scale
        new_section[6:] = optimized_section[6:]*inlet_length_scale
        section_diff = np.abs(new_section[0:-1] - new_section[1:])
        # radius_diff = outlet_radius_scale[0:5] -  outlet_radius_scale[1:6]   
        # radius_first = (radius_diff[0:3] -  radius_diff[1:4])/(section_diff[0:3])
        new_radius = copy.copy(optimized_radius)
        # new_radius[0] *=outlet_radius_scale[0]
        new_radius[1] = new_radius[0]*outlet_radius_scale[1]
        new_radius[2] = new_radius[1]*outlet_radius_scale[2]
        new_radius[3] = new_radius[2]*outlet_radius_scale[3]

        # print("radius: ", new_radius)

        radius_diff = (new_radius[0:6] - new_radius[1:7])/(section_diff[0:6])
        radius_first = (radius_diff[0:5] -  radius_diff[1:6])

        # print("curvature: ", radius_first)

        outputs["curvature"] = radius_first


