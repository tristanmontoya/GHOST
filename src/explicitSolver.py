# Tristan Montoya - Euler 1D - Temporal Discretization and Iterative Methods

import numpy as np
import scipy.linalg as la

class explicitSolver:

    def __init__(self, runName, spatialDiscretization, C, alpha=[1./4., 1./6., 3./8., 0.5, 1.0], max_its = 1000,
                 useLocalTimeStep=False, implicitResidualSmoothing=False, beta=0.6, gamma_3 = 0.56, gamma_5 = 0.44,
                 multiGrid=False, n_grids=1, ref_u = 300, ref_a = 315, rel_tol = 1.e-12):

        #problem setup
        self.runName = runName
        self.spatialDiscretization = spatialDiscretization
        self.C = C #Courant number
        self.useLocalTimeStep = useLocalTimeStep
        self.rel_tol = rel_tol
        self.alpha = np.copy(alpha)
        self.ref_u = ref_u
        self.ref_a = ref_a
        self.max_its = max_its
        self.multiGrid = multiGrid
        self.n_grids = n_grids
        self.q = len(alpha) #number of stages
        self.gamma_3 = gamma_3
        self.gamma_5 = gamma_5

        self.implicitResidualSmoothing = implicitResidualSmoothing
        self.beta=beta

        #initialization
        self.t = 0.0 #initialize time at 0.0
        self.n = 0 # no time steps taken
        self.Q = spatialDiscretization.Q_0
        self.resHistory = []


    #update matrix of local time steps (not the inverse, this will go on the RHS, it's not an implicit method!)
    #call for each step that's taken
    def setTimeStepMatrix(self, Q):

        self.Tmat = np.zeros(shape=[3*(self.spatialDiscretization.M), 3*self.spatialDiscretization.M])

        if self.useLocalTimeStep == True:
            for i in range(0, self.spatialDiscretization.M):

                dt = self.C * self.spatialDiscretization.dx / \
                     (self.spatialDiscretization.problem.specrA_j(Q[i * 3:(i + 1) * 3],
                                                                  self.spatialDiscretization.mesh[i]))
                self.Tmat[(i*3):(i+1)*3, (i*3):(i+1)*3] = dt*np.eye(3)
        else:
            self.dt = self.C * self.spatialDiscretization.dx / (self.ref_u + self.ref_a)
            self.Tmat = np.eye(self.spatialDiscretization.M*3)*self.dt


    def runSolver(self):

        if self.multiGrid==False:

            self.R = -1.0*self.spatialDiscretization.buildFlowResidual(self.Q)
            res = np.linalg.norm(self.R)
            res_0 = res
            self.resHistory.append(res)

            while (self.n < self.max_its and res/res_0 > self.rel_tol):
                self.Q, self.R = self.runStep(self.Q, forcingTerm=np.zeros(self.spatialDiscretization.M*3))
                res = np.linalg.norm(self.R)
                self.resHistory.append(res)
                self.n = self.n + 1
                print("iteration ", self.n, "res norm = ", res, "rel_res = ",res/res_0)

        self.saveResidualHistory()
        self.saveResults()

    def runStep(self, Q, forcingTerm):
        #make sure if using multigrid, that setMeshMultiGrid has been called

        #update local time steps based on current solution (might want to do this at every stage)
        self.setTimeStepMatrix(Q)

        Q_0 = np.copy(Q)
        R_diss = [np.zeros(self.spatialDiscretization.M*3), #0
                  np.zeros(self.spatialDiscretization.M*3), #2
                  np.zeros(self.spatialDiscretization.M*3)] #4

        # stage 1
        Q_inlet, Q_exit = self.spatialDiscretization.getBoundaryData(Q)
        R_inviscid = self.spatialDiscretization.buildFlowResidualInviscid(Q,Q_inlet,Q_exit)
        self.spatialDiscretization.setDissCoeff(Q,Q_inlet,Q_exit)
        R_diss[0] = self.spatialDiscretization.buildD2Residual(Q,Q_inlet,Q_exit) + \
            self.spatialDiscretization.buildD4Residual(Q,Q_inlet,Q_exit)
        Rdt = self.Tmat @ (-1.0*(R_inviscid + 1.0*R_diss[0]) + forcingTerm)

        if self.implicitResidualSmoothing:
            Rdt = self.applyResidualSmoothing(Rdt)

        Q = Q_0 - self.alpha[0]*Rdt

        # stage 2
        Q_inlet, Q_exit = self.spatialDiscretization.getBoundaryData(Q)
        R_inviscid = self.spatialDiscretization.buildFlowResidualInviscid(Q,Q_inlet,Q_exit)
        Rdt = self.Tmat @ (-1.0*(R_inviscid + 1.0 * R_diss[0])  + forcingTerm)

        if self.implicitResidualSmoothing:
            Rdt = self.applyResidualSmoothing(Rdt)

        Q = Q_0 - self.alpha[1]*Rdt

        # stage 3
        Q_inlet, Q_exit = self.spatialDiscretization.getBoundaryData(Q)
        R_inviscid = self.spatialDiscretization.buildFlowResidualInviscid(Q, Q_inlet, Q_exit)
        self.spatialDiscretization.setDissCoeff(Q, Q_inlet, Q_exit)
        R_diss[1] = self.spatialDiscretization.buildD2Residual(Q,Q_inlet,Q_exit) + \
            self.spatialDiscretization.buildD4Residual(Q,Q_inlet,Q_exit)
        Rdt = self.Tmat @ (-1.0*(R_inviscid + (1.0 - self.gamma_3) * R_diss[0] + self.gamma_3 * R_diss[1]) + \
                           forcingTerm)

        if self.implicitResidualSmoothing:
            Rdt = self.applyResidualSmoothing(Rdt)

        Q = Q_0 - self.alpha[2]* Rdt

        # stage 4
        Q_inlet, Q_exit = self.spatialDiscretization.getBoundaryData(Q)
        R_inviscid = self.spatialDiscretization.buildFlowResidualInviscid(Q, Q_inlet, Q_exit)
        Rdt = self.Tmat @ (-1.0*(R_inviscid + (1.0 - self.gamma_3) * R_diss[0] + self.gamma_3 * R_diss[1]) + forcingTerm)

        if self.implicitResidualSmoothing:
            Rdt = self.applyResidualSmoothing(Rdt)

        Q = Q_0 - self.alpha[3] * Rdt

        #stage 5
        Q_inlet, Q_exit = self.spatialDiscretization.getBoundaryData(Q)
        R_inviscid = self.spatialDiscretization.buildFlowResidualInviscid(Q, Q_inlet, Q_exit)
        self.spatialDiscretization.setDissCoeff(Q, Q_inlet, Q_exit)
        R_diss[2] = self.spatialDiscretization.buildD2Residual(Q,Q_inlet,Q_exit) + \
            self.spatialDiscretization.buildD4Residual(Q,Q_inlet,Q_exit)
        Rdt = self.Tmat @ (-1.0*(R_inviscid + (1.0 - self.gamma_3) * (1.0 - self.gamma_5) * R_diss[0] + \
            self.gamma_3 * (1.0 - self.gamma_5)*R_diss[1] + self.gamma_5*R_diss[2]) + forcingTerm)

        if self.implicitResidualSmoothing:
            Rdt = self.applyResidualSmoothing(Rdt)

        Q = Q_0 - self.alpha[4]*Rdt

        #return residual
        R = -1.0*self.spatialDiscretization.buildFlowResidual(Q) + forcingTerm

        return Q, R

    #Takes in the residual but multiplied by local timestep
    def applyResidualSmoothing(self, Rdt):
        Rdt_smoothed = np.zeros(self.spatialDiscretization.M*3)

        Rdt_mass = Rdt[0::3]
        Rdt_mom = Rdt[1::3]
        Rdt_ene = Rdt[2::3]

        smoother = np.zeros((3,self.spatialDiscretization.M))
        smoother[0] = -1.0*self.beta
        smoother[1] = 1.0 + 2.0*self.beta
        smoother[2] = -1.0*self.beta


        Rdt_smoothed[0::3] = la.solve_banded((1,1),smoother,Rdt_mass)
        Rdt_smoothed[1::3] = la.solve_banded((1,1),smoother,Rdt_mom)
        Rdt_smoothed[2::3] = la.solve_banded((1,1),smoother,Rdt_ene)

        return Rdt_smoothed

    def saveResidualHistory(self):
        resHistory = np.array([np.arange(0,len(self.resHistory)), self.resHistory])
        np.save("../results/"+self.runName+"_resHistory.npy", resHistory)

    def saveResults(self):
        Q1 = self.Q[0::3] #0
        Q2 = self.Q[1::3] #1
        Q3 = self.Q[2::3] #2
        rho = np.zeros(self.spatialDiscretization.M) #3
        u = np.zeros(self.spatialDiscretization.M) #4
        e = np.zeros(self.spatialDiscretization.M) #5
        p = np.zeros(self.spatialDiscretization.M) #6
        a = np.zeros(self.spatialDiscretization.M) #7
        Ma = np.zeros(self.spatialDiscretization.M) #8

        #9 is mesh

        for i in range(0, self.spatialDiscretization.M):

            rho[i] = Q1[i]/self.spatialDiscretization.problem.S(self.spatialDiscretization.mesh[i])
            u[i] = Q2[i]/Q1[i]
            e[i] = Q3[i]/self.spatialDiscretization.problem.S(self.spatialDiscretization.mesh[i])
            p[i] = (self.spatialDiscretization.problem.gamma - 1.) * (e[i] - 0.5 * rho[i]*u[i]**2)
            a[i] = np.sqrt(self.spatialDiscretization.problem.gamma*p[i]/rho[i])
            Ma[i] = u[i]/a[i]

        np.save("../results/"+self.runName+"_results.npy", np.array([Q1, Q2, Q3,
                                                      rho, u, e, p,
                                                       a, Ma, self.spatialDiscretization.mesh,
                                                                     self.spatialDiscretization.epsilon_2,
                                                                     self.spatialDiscretization.epsilon_4]))
