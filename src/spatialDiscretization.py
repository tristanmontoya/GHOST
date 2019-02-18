# Tristan Montoya - Euler 1D - Spatial Discretization

# High-order code and this will inherit from the same base class
# Takes in a Problem object, gets passed to temporalDiscretization which solves the resulting ODE

import numpy as np
import scipy as sp

class SpatialDiscretization:

    def __init__(self, problem, M, k_2, k_4):
        # M number of interior nodes
        self.problem = problem
        self.M = M
        self.k_2 = k_2 #second-difference dissipation
        self.k_4 = k_4 #fourth-difference dissipation
        self.meshGen()
        self.evaluateInitialConditionOnMesh()

    # equispaced grid not including boundary points
    def meshGen(self):
        grid_with_boundaries = np.linspace(0., self.problem.length, num=(self.M + 2))
        self.mesh = grid_with_boundaries[1:self.M+1]
        self.dx = self.problem.length/(self.M+1.)

    def evaluateInitialConditionOnMesh(self):
        self.Q_0 = np.ones(3*self.M)
        for i in range(0,self.M):
            self.Q_0[i*3:(i+1)*3] = self.problem.evaluateInitialCondition(self.mesh[i])

    def setInitialConditionOnMesh(self, Q_0):
        self.Q_0 = Q_0

    # difference operator applied to flux. Left argument first.
    def delta_E_j(self, Q_jm1, Q_jp1):
        return 1./(2.*self.dx)*(self.problem.E_j(Q_jp1) - self.problem.E_j(Q_jm1))

    # specify/extrapolate BCs- assuming all subsonic, zeroth order extrapolation
    def getBoundaryData(self, Q):

        if self.problem.bcMode == 0:
            #rho, rho*u at inlet, p at exit

            Q_inlet = np.zeros(3)
            Q_exit = np.zeros(3)

            #inlet specify rho and rho u, extrapolate p

            Q_inlet[0] = self.problem.rho_inlet*self.problem.S(0.)
            Q_inlet[1] = self.problem.rhou_inlet*self.problem.S(0.)

            p_extrap = (self.problem.gamma - 1.)*((Q[2]/self.problem.S(self.mesh[0])) \
                                                  - 0.5*Q[1]**2/(Q[0]*self.problem.S(self.mesh[0])))

            Q_inlet[2] = (p_extrap/(self.problem.gamma - 1.) + \
                          0.5*Q_inlet[1]**2/(Q_inlet[0]*self.problem.S(0.)))*self.problem.S(0.)

            #exit specify p, extrapolate rho, rhou

            Q_exit[0] = Q[(self.M*3)-3]*self.problem.S(self.problem.length)/self.problem.S(self.mesh[self.M-1])
            Q_exit[1] = Q[(self.M * 3) - 2] * self.problem.S(self.problem.length) / self.problem.S(self.mesh[self.M - 1])

            Q_exit[2] = (self.problem.p_exit / (self.problem.gamma - 1.) + \
                          0.5 * Q_exit[1] ** 2 / (Q_exit[0] * self.problem.S(self.problem.length))) * self.problem.S(self.problem.length)

        if self.problem.bcMode == 1:
            #full Q specified

            Q_inlet = self.problem.Q_in
            Q_exit = self.problem.Q_out

        return Q_inlet, Q_exit

    # Build R(Q) for dQ/dt = R(Q) + D(Q), D(Q) dissipation
    def buildFlowResidual(self, Q, updateDissipation=True):
        M = self.M
        R = np.zeros(M * 3)

        #interior points not including first and last (here indexed 0 and M-1)
        for j in range(1, M-1):
            R[j*3:(j+1)*3] = -self.delta_E_j(Q[(j-1)*3:j*3], Q[(j+1)*3:(j+2)*3]) \
                   + self.problem.H_j(Q[j*3:(j+1)*3],self.mesh[j])

        #get boundary condition information (must be updated based on Q)
        Q_inlet, Q_exit = self.getBoundaryData(Q)

        #inlet
        R[0:3] = -self.delta_E_j(Q_inlet, Q[3:6]) + self.problem.H_j(Q[0:3], self.mesh[0])

        #exit
        R[(M-1)*3:(M)*3] = -self.delta_E_j(Q[(M-2)*3:(M-1)*3], Q_exit) + self.problem.H_j(Q[(M-1)*3:M*3], self.mesh[M-1])

        #update dissipation coefficients
        if updateDissipation:
            self.setDissCoeff(Q,Q_inlet,Q_exit)

        # apply dissipation
        R = R + self.buildD2Residual(Q, Q_inlet, Q_exit) + self.buildD4Residual(Q, Q_inlet, Q_exit)

        return R

    # update (epsilon^(2)*sigma)_j+1/2 and (epsilon^(4)*sigma)_j+1/2 based on flow solution

    def setDissCoeff(self, Q, Q_inlet, Q_exit):

        self.D2coeff_jphalf = np.zeros(self.M - 1)
        self.D4coeff_jphalf = np.zeros(self.M - 1)

        # inlet
        sigma_inlet = self.problem.specrA_j(Q_inlet, 0.)
        sigma_0 = self.problem.specrA_j(Q[0:3], self.mesh[0])
        epsilon_2_inlet = self.k_2
        epsilon_2_0 = self.k_2
        epsilon_4_inlet = self.k_2
        epsilon_4_0 = self.k_2
        self.D2coeff_inlet = 0.5 * (epsilon_2_inlet * sigma_inlet * self.problem.S(0.) + \
                                    epsilon_2_0 * sigma_0 * self.problem.S(self.mesh[0]))
        self.D4coeff_inlet = 0.5 * (epsilon_4_inlet * sigma_inlet * self.problem.S(0.) + \
                                    epsilon_4_0 * sigma_0 * self.problem.S(self.mesh[0]))

        # exit
        sigma_Mm1 = self.problem.specrA_j(Q[(self.M - 1) * 3:(self.M) * 3], self.mesh[self.M - 1])
        sigma_exit = self.problem.specrA_j(Q_exit, self.problem.length)
        epsilon_2_Mm1 = self.k_2
        epsilon_2_exit = self.k_2
        epsilon_4_Mm1 = self.k_4
        epsilon_4_exit = self.k_4
        self.D2coeff_exit = 0.5 * (epsilon_2_Mm1 * sigma_Mm1 * self.problem.S(self.mesh[self.M - 1]) + \
                                   epsilon_2_exit * sigma_exit * self.problem.S(self.problem.length))
        self.D4coeff_exit = 0.5 * (epsilon_4_Mm1 * sigma_Mm1 * self.problem.S(self.mesh[self.M - 1]) + \
                                   epsilon_4_exit * sigma_exit * self.problem.S(self.problem.length))

        # interior
        for j in range(0, self.M - 1):
            # get spectral radii
            sigma_j = self.problem.specrA_j(Q[j * 3:(j + 1) * 3], self.mesh[j])
            sigma_jp1 = self.problem.specrA_j(Q[(j + 1) * 3:(j + 2) * 3], self.mesh[j + 1])

            epsilon_2_j = self.k_2
            epsilon_2_jp1 = self.k_2
            epsilon_4_j = self.k_4
            epsilon_4_jp1 = self.k_4

            self.D2coeff_jphalf[j] = 0.5 * (epsilon_2_j * sigma_j * self.problem.S(self.mesh[j]) + \
                                            epsilon_2_jp1 * sigma_jp1 * self.problem.S(self.mesh[j + 1]))
            self.D4coeff_jphalf[j] = 0.5 * (epsilon_4_j * sigma_j * self.problem.S(self.mesh[j]) + \
                                            epsilon_4_jp1 * sigma_jp1 * self.problem.S(self.mesh[j + 1]))

    # second-difference (first-order) dissipation
    def buildD2Residual(self, Q, Q_inlet, Q_exit):
        D_2 = np.zeros(self.M * 3)

        # for each interior node
        for j in range(1, self.M - 1):
            # remove the area from Q so uniform flow is preserved
            q_jm1 = Q[(j - 1) * 3:(j) * 3] / self.problem.S(self.mesh[j - 1])
            q_j = Q[j * 3:(j + 1) * 3] / self.problem.S(self.mesh[j])
            q_jp1 = Q[(j + 1) * 3:(j + 2) * 3] / self.problem.S(self.mesh[j + 1])
            D_2[j * 3:(j + 1) * 3] = self.D2coeff_jphalf[j] * (q_jp1 - q_j) - self.D2coeff_jphalf[j - 1] * (q_j - q_jm1)

        # inlet
        q_inlet = Q_inlet / self.problem.S(0.)
        q_0 = Q[0:3] / self.problem.S(self.mesh[0])
        q_1 = Q[3:6] / self.problem.S(self.mesh[1])
        D_2[0:3] = self.D2coeff_jphalf[0] * (q_1 - q_0) - self.D2coeff_inlet * (q_0 - q_inlet)

        # exit
        q_Mm2 = Q[(self.M - 2) * 3:(self.M - 1) * 3] / self.problem.S(self.mesh[self.M - 2])
        q_Mm1 = Q[(self.M - 1) * 3:(self.M) * 3] / self.problem.S(self.mesh[self.M - 1])
        q_exit = Q_exit / self.problem.S(self.problem.length)
        D_2[(self.M - 1) * 3:(self.M) * 3] = self.D2coeff_exit * (q_exit - q_Mm1) - self.D2coeff_jphalf[self.M - 2] * (
                    q_Mm1 - q_Mm2)

        return D_2/self.dx

    # fourth-difference (third-order) dissipation
    def buildD4Residual(self, Q, Q_inlet, Q_exit):
        D_4 = np.zeros(self.M * 3)

        # for each node at more than two away from a boundary
        for j in range(2, self.M - 2):
            # remove the area from Q so uniform flow is preserved
            q_jm2 = Q[(j - 2) * 3:(j-1) * 3] / self.problem.S(self.mesh[j - 2])
            q_jm1 = Q[(j - 1) * 3:(j) * 3] / self.problem.S(self.mesh[j - 1])
            q_j = Q[j * 3:(j + 1) * 3] / self.problem.S(self.mesh[j])
            q_jp1 = Q[(j + 1) * 3:(j + 2) * 3] / self.problem.S(self.mesh[j + 1])
            q_jp2 = Q[(j + 2) * 3:(j + 3) * 3] / self.problem.S(self.mesh[j + 2])

            D_4[j * 3:(j + 1) * 3] = self.D4coeff_jphalf[j] * (q_jp2 - 3*q_jp1 + 3*q_j - q_jm1) - \
                                     self.D4coeff_jphalf[j - 1] * (q_jp1 - 3*q_j + 3*q_jm1 - q_jm2)

        # inlet
        q_inlet = Q_inlet / self.problem.S(0.)
        q_0 = Q[0:3] / self.problem.S(self.mesh[0])
        q_1 = Q[3:6] / self.problem.S(self.mesh[1])
        q_2 = Q[6:9] / self.problem.S(self.mesh[2])
        q_3 = Q[9:12] / self.problem.S(self.mesh[3])

        D_4[0:3] = self.D4coeff_jphalf[0] * (q_2 - 3 * q_1 + 3 * q_0 - q_inlet) - \
                   self.D4coeff_inlet * (q_1 - 2*q_0 + q_inlet)
        D_4[3:6] = self.D4coeff_jphalf[1] * (q_3 - 3 * q_2 + 3 * q_1 - q_0) - \
                                 self.D4coeff_jphalf[0] * (q_2 - 3 * q_1 + 3 * q_0 - q_inlet)


        # exit
        q_Mm4 = Q[(self.M - 4) * 3:(self.M - 3) * 3] / self.problem.S(self.mesh[self.M - 4])
        q_Mm3 = Q[(self.M - 3) * 3:(self.M - 2) * 3] / self.problem.S(self.mesh[self.M - 3])
        q_Mm2 = Q[(self.M - 2) * 3:(self.M - 1) * 3] / self.problem.S(self.mesh[self.M - 2])
        q_Mm1 = Q[(self.M - 1) * 3:(self.M) * 3] / self.problem.S(self.mesh[self.M - 1])
        q_exit = Q_exit / self.problem.S(self.problem.length)

        D_4[(self.M - 1) * 3:(self.M) * 3] = self.D4coeff_exit * (-q_exit + 2*q_Mm1 - q_Mm2) - \
                   self.D4coeff_jphalf[self.M-2] * (q_exit - 3*q_Mm1 + 3*q_Mm2 - q_Mm3)
        D_4[(self.M - 2) * 3:(self.M - 1) * 3] = self.D4coeff_jphalf[self.M-2] * (q_exit - 3*q_Mm1 + 3*q_Mm2 - q_Mm3) - \
                                 self.D4coeff_jphalf[self.M-3] * (q_Mm1  - 3 * q_Mm2  + 3 * q_Mm3 - q_Mm4)

        return -D_4/self.dx #negative sign needed here, since residual is on RHS


    # Build flow Jacobian dR/dQ for entire mesh
    def buildFlowJacobian(self, Q):
        dRdQ = np.zeros(shape=[3*self.M, 3*self.M])
        for i in range(0, self.M):
            #diagonal element
            dRdQ[i*3:(i+1)*3, i*3:(i+1)*3] = self.problem.B_j(Q[i*3:(i+1)*3], self.mesh[i])
            #sub-diagonal
            if i > 0:
                dRdQ[i * 3:(i + 1) * 3, (i-1) * 3:(i) * 3] = 1./(2.*self.dx)*self.problem.A_j(Q[(i-1)*3:(i)*3])
            #super-diagonal
            if i < self.M-1:
                dRdQ[i * 3:(i + 1) * 3, (i+1) * 3:(i+2) * 3] = -1./(2.*self.dx)*self.problem.A_j(Q[(i+1) * 3:(i+2) * 3])

        dRdQ = dRdQ + self.buildD2Jacobian() + self.buildD4Jacobian()
        return dRdQ

    # Build dD2/dQ for entire mesh
    def buildD2Jacobian(self):

        dD2dQ = np.zeros(shape=[3 * self.M, 3 * self.M])

        #interior
        for i in range(1, self.M-1):
            # diagonal element
            dD2dQ[i * 3:(i + 1) * 3, i * 3:(i + 1) * 3] = -1.0*(self.D2coeff_jphalf[i] + self.D2coeff_jphalf[i-1])/self.problem.S(self.mesh[i])*np.eye(3)
            # sub-diagonal
            dD2dQ[i * 3:(i + 1) * 3, (i - 1) * 3:i * 3] = self.D2coeff_jphalf[i-1]/self.problem.S(self.mesh[i-1])*np.eye(3)
            # super-diagonal
            dD2dQ[i * 3:(i + 1) * 3, (i + 1) * 3:(i + 2) * 3] = self.D2coeff_jphalf[i]/self.problem.S(self.mesh[i+1])*np.eye(3)

        #inlet (0,0)
        dD2dQ[0:3, 0:3] = -1.0 * (self.D2coeff_jphalf[0] + self.D2coeff_inlet)/self.problem.S(self.mesh[0])* np.eye(3)
        #inlet (0, 1)
        dD2dQ[0:3, 3:6] = self.D2coeff_jphalf[0]/self.problem.S(self.mesh[1])*np.eye(3)

        #exit (M-1, M-1)
        dD2dQ[(self.M-1) * 3:(self.M)* 3, (self.M-1)* 3:(self.M) * 3] = \
            -1.0 * (self.D2coeff_jphalf[self.M-2] + self.D2coeff_exit)/self.problem.S(self.mesh[self.M-1])*np.eye(3)
        #exit (M-1, M-2)
        dD2dQ[(self.M - 1) * 3:(self.M) * 3, (self.M - 2) * 3:(self.M-1) * 3] = self.D2coeff_jphalf[self.M-2]/self.problem.S(self.mesh[self.M-2])*np.eye(3)

        return dD2dQ/self.dx

    # Build dD2/dQ for entire mesh
    # (again constructing the negative and then flipping the sign like I did with the D4 residual)
    def buildD4Jacobian(self):

        dD4dQ = np.zeros(shape=[3 * self.M, 3 * self.M])

        # interior
        for i in range(2, self.M - 2):
            # diagonal element
            dD4dQ [i * 3:(i + 1) * 3, i * 3:(i + 1) * 3] = 3.0 * ( \
                        self.D4coeff_jphalf[i] + self.D4coeff_jphalf[i - 1]) / self.problem.S(self.mesh[i]) * np.eye(3)
            # sub-diagonal
            dD4dQ[i * 3:(i + 1) * 3, (i - 1) * 3:i * 3] = (-1.0*self.D4coeff_jphalf[i] - 3.0*self.D4coeff_jphalf[i-1]) / self.problem.S(
                self.mesh[i - 1]) * np.eye(3)
            dD4dQ[i * 3:(i + 1) * 3, (i - 2) * 3:(i-1) * 3] = self.D4coeff_jphalf[i-1] / self.problem.S(self.mesh[i - 2]) * np.eye(3)

            # super-diagonal
            dD4dQ[i * 3:(i + 1) * 3, (i + 1) * 3:(i + 2) * 3] = \
                (-3.0*self.D4coeff_jphalf[i] - 1.0*self.D4coeff_jphalf[i-1]) / self.problem.S(self.mesh[i + 1]) * np.eye(3)
            dD4dQ[i * 3:(i + 1) * 3, (i + 2) * 3:(i + 3) * 3] = \
                self.D4coeff_jphalf[i] / self.problem.S(self.mesh[i + 2]) *np.eye(3)

        # inlet (0,0)
        dD4dQ[0:3, 0:3] = (3.*self.D4coeff_jphalf[0] + 2.*self.D4coeff_inlet) / self.problem.S(self.mesh[0]) * np.eye(3)
        # inlet (0, 1)
        dD4dQ[0:3, 3:6] = (-3.*self.D4coeff_jphalf[0] - 1.*self.D4coeff_inlet) / self.problem.S(self.mesh[1]) * np.eye(3)
        # inlet (0, 2)
        dD4dQ[0:3, 6:9] = self.D4coeff_jphalf[0]/ self.problem.S(self.mesh[2]) * np.eye(3)
        # inlet (1, 0)
        dD4dQ[3:6, 0:3] = (-1.0*self.D4coeff_jphalf[1] - 3.0*self.D4coeff_jphalf[0]) / self.problem.S(self.mesh[0]) * np.eye(3)
        # inlet (1, 1)
        dD4dQ[3:6, 3:6] = 3.0 * ( self.D4coeff_jphalf[1] + self.D4coeff_jphalf[0]) / self.problem.S(self.mesh[1]) * np.eye(3)
        # inlet (1, 2)
        dD4dQ[3:6, 6:9] = (-3.0 * self.D4coeff_jphalf[1] - 1.0 * self.D4coeff_jphalf[0]) / self.problem.S(self.mesh[2]) * np.eye(3)
        # inlet (1, 3)
        dD4dQ[3:6, 9:12] = self.D4coeff_jphalf[1] / self.problem.S(self.mesh[3]) * np.eye(3)

        # exit (M-1, M-1)
        dD4dQ[(self.M - 1) * 3:(self.M) * 3, (self.M - 1) * 3:(self.M) * 3] = \
            (3.*self.D4coeff_jphalf[self.M - 2] + 2.*self.D4coeff_exit) / self.problem.S(self.mesh[self.M - 1]) * np.eye(3)
        # exit (M-1, M-2)
        dD4dQ[(self.M - 1) * 3:(self.M) * 3, (self.M - 2) * 3:(self.M - 1) * 3] = \
            (-3. * self.D4coeff_jphalf[self.M-2] - 1. * self.D4coeff_exit) / self.problem.S(self.mesh[self.M - 2]) * np.eye(3)
        # exit (M-1, M-3)
        dD4dQ[(self.M - 1) * 3:(self.M) * 3, (self.M - 3) * 3:(self.M - 2) * 3] = \
            self.D4coeff_jphalf[self.M-2] / self.problem.S(self.mesh[self.M -3]) * np.eye(3)
        # inlet (M-2, M-1)
        dD4dQ[(self.M - 2) * 3:(self.M - 1) * 3, (self.M - 1) * 3:(self.M ) * 3] = \
            (-3.0 * self.D4coeff_jphalf[self.M-2] - 1.0 * self.D4coeff_jphalf[self.M - 3]) / \
            self.problem.S(self.mesh[self.M-1]) * np.eye(3)
        # inlet (M-2, M-2)
        dD4dQ[(self.M - 2) * 3:(self.M - 1) * 3, (self.M - 2) * 3:(self.M - 1) * 3] = \
            3.0 * (self.D4coeff_jphalf[self.M - 2] + self.D4coeff_jphalf[self.M - 3]) / self.problem.S(self.mesh[self.M - 2]) * np.eye(3)
        # inlet (M-2, M-3)
        dD4dQ[(self.M - 2) * 3:(self.M - 1) * 3, (self.M - 3) * 3:(self.M - 2) * 3] = \
            (-1.0 * self.D4coeff_jphalf[self.M - 2] - 3.0 * self.D4coeff_jphalf[self.M - 3]) / self.problem.S(self.mesh[self.M - 3]) * np.eye(3)
        # inlet (M-2, M-4)
        dD4dQ[(self.M - 2) * 3:(self.M - 1) * 3, (self.M - 4) * 3:(self.M - 3) * 3] = \
            self.D4coeff_jphalf[self.M - 3] / self.problem.S(self.mesh[self.M - 4]) * np.eye(3)

        return -dD4dQ / self.dx

    # Verify that linearization is correct (finite difference)
    def runChecks(self, Q_0, Q_in, Q_out):

        # update dissipation coefficients, then hold fixed
        self.setDissCoeff(Q_0, Q_in, Q_out)

        print("Initial condition: \n", Q_0)

        print("test flow residual: \n", self.buildFlowResidual(Q_0))
        print("test dissipation 2 residual: \n", self.buildD2Residual(Q_0, Q_in, Q_out))
        print("test dissipation 4 residual: \n", self.buildD4Residual(Q_0, Q_in, Q_out))

        dRdQ = self.buildFlowJacobian(Q_0)
        dD2dQ = self.buildD2Jacobian()
        dD4dQ = self.buildD4Jacobian()

        print("Flow Jacobian: \n", dRdQ)
        FDFlow = self.testFlowJacobianFiniteDifference(Q_0, Q_in, Q_out)
        print("finite difference: \n", FDFlow)
        print("error: \n", dRdQ - FDFlow)

        print("Second-Difference Dissipation Jacobian: \n", dD2dQ)
        FDD2 = self.testD2JacobianFiniteDifference(Q_0, Q_in, Q_out)
        print("finite difference: \n", FDD2)
        print("error: \n", dD2dQ - FDD2)

        print("Fourth-Difference Dissipation Jacobian: \n", dD4dQ)
        FDD4 = self.testD4JacobianFiniteDifference(Q_0, Q_in, Q_out)
        print("finite difference: \n", FDD4)
        print("error: \n", dD4dQ - FDD4)

        print("Max error in D2 residual: ", np.amax(np.absolute(dD2dQ - FDD2)))
        print("Max error in D4 residual: ", np.amax(np.absolute(dD4dQ - FDD4)))
        print("Max error in flow residual: ", np.amax(np.absolute(dRdQ - FDFlow)))


    def testD2JacobianFiniteDifference(self, Q, Q_in, Q_out):
        residual = lambda Q, row : self.buildD2Residual(Q, Q_in, Q_out)[row]
        jacobian = np.zeros(shape=[self.M*3, self.M*3])
        for i in range(0, self.M*3):
            jacobian[i,:] = sp.optimize.approx_fprime(Q, residual, np.sqrt(np.finfo(np.float).eps), i)

        return jacobian

    def testD4JacobianFiniteDifference(self, Q, Q_in, Q_out):
        residual = lambda Q, row: self.buildD4Residual(Q, Q_in, Q_out)[row]
        jacobian = np.zeros(shape=[self.M * 3, self.M * 3])
        for i in range(0, self.M * 3):
            jacobian[i, :] = sp.optimize.approx_fprime(Q, residual, np.sqrt(np.finfo(np.float).eps), i)

        return jacobian

    def testFlowJacobianFiniteDifference(self, Q, Q_in, Q_out):
        self.setDissCoeff(Q, Q_in, Q_out)
        residual = lambda Q, row : self.buildFlowResidual(Q, updateDissipation=False)[row]
        jacobian = np.zeros(shape=[self.M*3, self.M*3])
        for i in range(0, self.M*3):
            jacobian[i,:] = sp.optimize.approx_fprime(Q, residual, np.sqrt(np.finfo(np.float).eps), i)
        return jacobian
