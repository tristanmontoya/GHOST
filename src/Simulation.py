# GHOST - Top-Level Simulation Object

class Simulation:

    """

    # Properties

    problem (Problem)
        # The physical aspects of the problem, i.e. PDE, BC, IC, split-form, numerical fluxes

    mesh (AffineMesh)
        # the mesh (does not include any numerics associated with it)

    local_discretization (LocalDiscretizationBase)
        # numerics for specified reference element type for the problem
        # currently will assume all elements are mapped from the same reference element

    time_discretization (TimeDiscretization)
        # information about the (for now explicit) temporal scheme
        # includes stuff like time step size (calculates here), number of stages, etc.
        # does not actually take the step, as this requires evaluateResidual, which depends on problem, mesh, and
          local_discretization

    # Methods

    evaluate_residual
    time_step

    """