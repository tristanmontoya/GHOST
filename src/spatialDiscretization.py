# Tristan Montoya - Euler 1D - Spatial Discretization
# Maybe make this inherit from some general spatial discretization class

import numpy as np

class SpatialDiscretization:

    def __init__(self, problem, M, useDiss=False):
        # M number of interior nodes
        self.problem = problem
        self.M = M
        self.useDiss = useDiss
        self.meshGen()

    def meshGen(self):
        grid_with_boundaries np.linspace(0., self.problem.length, num=(M + 2))
        self.mesh = grid_withboundaries[1:M+1]
        self.dx = self.problem.length/(M+1.)

    def getBoundaryData(self, Q):
        Q_inlet = np.zeros(3)
        Q_exit = np.zeros(3)

        #specify/extrapolate (zeroth order) BCs - assuming all subsonic

        #inlet specify rho and rho u, extrapolate p

        Q_inlet[0] = self.problem.rho_inlet*self.problem.S(0.)
        Q_inlet[1] = self.problem.rhou_inlet*self.problem.S(0.)

        p_extrap = (self.problem.gamma - 1.)*((Q[2]/self.problem.S(self.mesh[0])) \
                                              - 0.5*Q[1]**2/(Q[0]*self.problem.S(self.mesh[0])))

        Q_inlet[2] = (p_extrap/(self.problem.gamma - 1.) + \
                      0.5*Q_inlet[1]**2/(Q_inlet[0]*self.problem.S(0.)))*self.problem.S(0.)

        #exit specify p, extrapolate rho, rhou

        Q_exit[0] = Q[(self.M*3)-3]*self.problem.S(self.problem.length)/self.problem.S(self.mesh[M-1])
        Q_exit[1] = Q[(self.M * 3) - 2] * self.problem.S(self.problem.length) / self.problem.S(self.mesh[M - 1])

        Q_exit[2] = (self.problem.p_exit / (self.problem.gamma - 1.) + \
                      0.5 * Q_exit[1] ** 2 / (Q_exit[0] * self.problem.S(self.problem.length))) * self.problem.S(self.problem.length)

        return Q_inlet, Q_exit

    # Build R(Q) for dQ/dt = R(Q) + D(Q), D(Q) dissipation
    def buildFlowResidual(self, Q):

        #interior points not including first and last (here indexed 0 and M-1)
        R = np.zeros(M)
        for j in range(1, M-1):
            R[j] = -self.delta_E_j(Q[(j-1)*3:j*3], Q[(j+1)*3:(j+2)*3]) \
                   + self.problem.H_j(Q[j*3:(j+1)*3],self.mesh[j]) #no dissipation for now

        #get boundary condition information (must be updated based on Q)
        Q_inlet, Q_exit = self.getBoundaryData(Q)

        #inlet
        R[0] = -self.delta_E_j(Q_inlet, Q[3:6]) + self.delta_E_j(Q[0:3], self.mesh[0])

        #exit
        R[M-1] = -self.delta_E_j(Q[(M-2)*3:(M-1)*3], Q_exit) + self.delta_E_j(Q[(M-1)*3:M*3], self.mesh[M-1])

        return R

    # Build flow Jacobian for entire mesh dR/dQ with no dissipation
    def buildFlowJacobian(self, Q):

        return J

    def delta_E_j(self, Q_jm1, Q_jp1):
        # difference operator applied to flux
        return 1./self.dx*(self.problem.E_j(Q_jp1) - self.problem.E_j(Q_jm1))



