# GHOST - Top-Level Simulation Object

class Simulation:

    """

    # Properties

    physics (Physics)
        # The physical aspects of the problem, i.e. PDE, BC, IC, split-form, numerical fluxes

    mesh (AffineMesh)
        # the mesh (does not include any numerics associated with it)

    refNumerics (Element)
        # numerics for specified reference element type for the problem
        # currently will assume all elements are mapped from the same reference element

    timeDiscretization (TimeDiscretization)
        # information about the (for now explicit) temporal scheme
        # includes stuff like time step size (calculates here), number of stages, etc.
        # does not actually take the step, as this requires evaluateResidual, which depends on physics, mesh, and
        refNumerics

    # Methods

    evaluateResidual
    timeStep

    """