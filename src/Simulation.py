import local_discretization
import numpy as np


class Simulation:
    """
    Assembles the solver components to run the simulation

    Attributes
    ----------

    problem : Problem
        The physical aspects of the problem, i.e. PDE, BC, IC, split-form,
        numerical fluxes

    mesh : Mesh
        The mesh (does not include any numerics associated with it)

    spatial_discretization : SpatialDiscretization
        Properties of the global spatial discretization scheme. Uses the
         matrices from spatial_discretization.local_numerics to build the
        residual. contains the function evaluateResidual which gets called by the
        time_discretization function

    time_discretization : TimeDiscretization
        Information about the (for now explicit) temporal scheme
        Includes stuff like time step size (calculates here), number of stages, etc.
        Methods for taking the time step (pass in spatialDiscretization.EvaluateResidual)

    """
    def __init__(self, problem, mesh, spatial_discretization, time_discretization):

        return
