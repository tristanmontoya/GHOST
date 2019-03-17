# Tristan Montoya - Euler 1D - Temporal Discretization and Iterative Methods

import numpy as np
import scipy.linalg as la

class explicitSolver:

    def __init__(self, runName, spatialDiscretization, C, alpha=np.array([1./4., 1./6., 3./8., 0.5, 1.0]), max_its = 1000,
                 useLocalTimeStep=False, updateFullResidual=False, implicitResidualSmoothing=False, beta=0.6, gamma_3 = 0.56, gamma_5 = 0.44,
                 multiGrid=False, n_grids=2, ref_u = 300, ref_a = 315, rel_tol = 1.e-12):

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
        self.gamma_3 = gamma_3
        self.gamma_5 = gamma_5

        self.updateFullResidual = updateFullResidual
        self.implicitResidualSmoothing = implicitResidualSmoothing
        self.beta=beta

        #initialization
        self.t = 0.0 #initialize time at 0.0
        self.n = 0 # no time steps taken
        self.Q = spatialDiscretization.Q_0
        self.resHistory = []
        if self.multiGrid:
            self.spatialDiscretization.meshGenMultigrid(self.n_grids)


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
        # results = np.load("../results/subsonic_C7_irsB0.6_smoothbounds_linear_results.npy")
        # self.Q[0::3] = results[0]
        # self.Q[1::3] = results[1]
        # self.Q[2::3] = results[2]

        self.R = -1.0*self.spatialDiscretization.buildFlowResidual(self.Q)
        #print("Initial R:\n:", self.R)
        res = np.linalg.norm(self.R[0::3])
        res_0 = res
        self.resHistory.append(res)

        while (self.n < self.max_its and res/res_0 > self.rel_tol):
            if self.multiGrid:
                self.Q = self.recursiveMGCycle(self.Q, 1, 0, forcingTerm=np.zeros(self.spatialDiscretization.M*3))
            else:
                self.Q = self.runStep(self.Q, forcingTerm=np.zeros(self.spatialDiscretization.M*3))
            self.R = self.spatialDiscretization.buildFlowResidual(self.Q)
            #print("R: \n", self.R)
            res = np.linalg.norm(self.R[0::3])
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
        if self.updateFullResidual:
            Rdt = self.Tmat @ (-1.0 * self.spatialDiscretization.buildFlowResidual(Q) + forcingTerm)
        else:
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
        if self.updateFullResidual:
            Rdt = self.Tmat @ (-1.0 * self.spatialDiscretization.buildFlowResidual(Q)+ forcingTerm)
        else:
            Q_inlet, Q_exit = self.spatialDiscretization.getBoundaryData(Q)
            R_inviscid = self.spatialDiscretization.buildFlowResidualInviscid(Q,Q_inlet,Q_exit)
            Rdt = self.Tmat @ (-1.0*(R_inviscid + 1.0 * R_diss[0]) + forcingTerm)
        if self.implicitResidualSmoothing:
            Rdt = self.applyResidualSmoothing(Rdt)

        Q = Q_0 - self.alpha[1]*Rdt

        # stage 3
        if self.updateFullResidual:
            Rdt = self.Tmat @ (-1.0 * self.spatialDiscretization.buildFlowResidual(Q) + forcingTerm)
        else:
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
        if self.updateFullResidual:
            Rdt = self.Tmat @ (-1.0 * self.spatialDiscretization.buildFlowResidual(Q) + forcingTerm)
        else:
            Q_inlet, Q_exit = self.spatialDiscretization.getBoundaryData(Q)
            R_inviscid = self.spatialDiscretization.buildFlowResidualInviscid(Q, Q_inlet, Q_exit)
            Rdt = self.Tmat @ (-1.0*(R_inviscid + (1.0 - self.gamma_3) * R_diss[0] + self.gamma_3 * R_diss[1]) + forcingTerm)
        if self.implicitResidualSmoothing:
            Rdt = self.applyResidualSmoothing(Rdt)

        Q = Q_0 - self.alpha[3] * Rdt

        #stage 5
        if self.updateFullResidual:
            Rdt = self.Tmat @ (-1.0 * self.spatialDiscretization.buildFlowResidual(Q) + forcingTerm)
        else:
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

        return Q

    #Takes in the residual but multiplied by local timestep
    def applyResidualSmoothing(self, Rdt, smoothBoundaries=True):
        Rdt_smoothed = np.zeros(self.spatialDiscretization.M*3)

        Rdt_mass = Rdt[0::3]
        Rdt_mom = Rdt[1::3]
        Rdt_ene = Rdt[2::3]

        smoother = np.zeros((3,self.spatialDiscretization.M))
        smoother[0] = -1.0*self.beta
        smoother[1] = 1.0 + 2.0*self.beta
        smoother[2] = -1.0*self.beta

        # if specified not to, don't smooth nodes next to boundaires
        if smoothBoundaries==False:
            smoother[0,1] = 0.0
            smoother[1,0] = 1.0
            smoother[2,self.spatialDiscretization.M - 2] = 0.0
            smoother[1,self.spatialDiscretization.M-1] = 1.0

        Rdt_smoothed[0::3] = la.solve_banded((1,1),smoother,Rdt_mass)
        Rdt_smoothed[1::3] = la.solve_banded((1,1),smoother,Rdt_mom)
        Rdt_smoothed[2::3] = la.solve_banded((1,1),smoother,Rdt_ene)

        return Rdt_smoothed

    def runTwoGridCycle(self, Q):
        Q = self.runStep(Q, np.zeros(self.spatialDiscretization.M*3))
        R = -1.0*self.spatialDiscretization.buildFlowResidual(Q)
        # print("initial:\n", Q)
        Q_0_restricted, R_restricted = self.restrict(Q,R, 1)
        Q_0_inlet, Q_0_exit = self.spatialDiscretization.getBoundaryData(Q_0_restricted)
        # print("Restricted Residual\n", R_restricted)
        # print("Restricted:\n", Q_0_restricted)
        #plus because of residual sign convention
        forcingTerm = R_restricted + self.spatialDiscretization.buildFlowResidual(Q_0_restricted)
        Q_corrected = self.runStep(Q_0_restricted, forcingTerm)

        Q_inlet, Q_exit = self.spatialDiscretization.getBoundaryData(Q_corrected)
        # print("Solved coarse:\n", Q_restricted)
        Q = Q + self.prolong(Q_corrected - Q_0_restricted, Q_inlet-Q_0_inlet, Q_exit-Q_0_exit, 0)
        # print("corrected:\n", Q)
        return Q

    def recursiveMGCycle(self, Q, reps, grid_level, forcingTerm): #set reps to 1 for V and 2 for w
        if grid_level == self.n_grids - 1:
            Q = self.runStep(Q, forcingTerm)
        else:
            Q = self.runStep(Q, forcingTerm)
            R = -1.0 * self.spatialDiscretization.buildFlowResidual(Q) + forcingTerm
            Q_0_restricted, R_restricted = self.restrict(Q, R, grid_level+1)
            Q_0_inlet, Q_0_exit = self.spatialDiscretization.getBoundaryData(Q_0_restricted)
            forcingTerm_new = R_restricted - (-1.0*self.spatialDiscretization.buildFlowResidual(Q_0_restricted))

            Q_old = np.copy(Q_0_restricted)
            for i in range(0,reps): #reps=1 does V cycle, 2 does W cycle
                Q_corrected = self.recursiveMGCycle(Q_old,reps,grid_level+1,forcingTerm_new)
                Q_old = np.copy(Q_corrected)

            Q_inlet, Q_exit = self.spatialDiscretization.getBoundaryData(Q_corrected)
            #Q = Q + self.prolong(Q_corrected - Q_0_restricted, np.zeros(3), np.zeros(3), grid_level)
            Q = Q + self.prolong(Q_corrected - Q_0_restricted, Q_inlet - Q_0_inlet, Q_exit - Q_0_exit, grid_level)
            Q = self.runStep(Q, forcingTerm) #run iteration after prolonging?
        return Q

    def restrict(self, Q, R, grid_level): #grid level is new grid level

        #update mesh
        self.spatialDiscretization.setMeshMultigrid(grid_level)
        Q_restricted = np.zeros(self.spatialDiscretization.M*3)
        R_restricted = np.zeros(self.spatialDiscretization.M * 3)
        print("restricting ", len(Q)//3, " to ", self.spatialDiscretization.M)

        Q_mass = Q[0::3]
        Q_mom = Q[1::3]
        Q_ene = Q[2::3]

        R_mass = R[0::3]
        R_mom = R[1::3]
        R_ene = R[2::3]

        #inject solution
        Q_restricted[0::3] = Q_mass[1::2]
        Q_restricted[1::3] = Q_mom[1::2]
        Q_restricted[2::3] = Q_ene[1::2]

        #boundary values are updated in runStep()

        #linear weighted restriction of residual
        for i in range(0,self.spatialDiscretization.M):
            im1_fine = i*2
            i_fine = i*2 + 1
            ip1_fine = i*2+2

            # Q_restricted[i*3 + 0] = 0.25*Q_mass[im1_fine] + 0.5*Q_mass[i_fine] + 0.25*Q_mass[ip1_fine]
            # Q_restricted[i * 3 + 1] = 0.25 * Q_mom[im1_fine] + 0.5 * Q_mom[i_fine] + 0.25 * Q_mom[ip1_fine]
            # Q_restricted[i * 3 + 2] = 0.25 * Q_ene[im1_fine] + 0.5 * Q_ene[i_fine] + 0.25 * Q_ene[ip1_fine]

            R_restricted[i*3 + 0] = 0.25*R_mass[im1_fine] + 0.5*R_mass[i_fine] + 0.25*R_mass[ip1_fine]
            R_restricted[i * 3 + 1] = 0.25 * R_mom[im1_fine] + 0.5 * R_mom[i_fine] + 0.25 * R_mom[ip1_fine]
            R_restricted[i * 3 + 2] = 0.25 * R_ene[im1_fine] + 0.5 * R_ene[i_fine] + 0.25 * R_ene[ip1_fine]

        return Q_restricted, R_restricted

    def prolong(self, Q, Q_inlet, Q_exit, grid_level): #grid level is new grid level
        # update mesh
        self.spatialDiscretization.setMeshMultigrid(grid_level)
        Q_prolonged = np.zeros(self.spatialDiscretization.M * 3)

        Q_mass = Q[0::3]
        Q_mom = Q[1::3]
        Q_ene = Q[2::3]

        print("prolonging ", len(Q)//3, " to ", self.spatialDiscretization.M)

        #linear weighted prolongation of solution
        for i in range(0,self.spatialDiscretization.M):
            if (i+1) % 2 == 0: #there exists a corresponding coarse grid node
                i_coarse = (i-1)//2
                #print("i: ", i, "i_coarse: ", i_coarse)

                Q_prolonged[i*3 + 0] = Q_mass[i_coarse]
                Q_prolonged[i * 3 + 1] = Q_mom[i_coarse]
                Q_prolonged[i * 3 + 2] = Q_ene[i_coarse]

            elif i == 0:
                Q_prolonged[i*3 + 0] = 0.5*Q_inlet[0] + 0.5*Q_mass[0]
                Q_prolonged[i * 3 + 1] =0.5*Q_inlet[1] + 0.5*Q_mom[0]
                Q_prolonged[i * 3 + 2] = 0.5*Q_inlet[2] +0.5*Q_ene[0]

            elif i == self.spatialDiscretization.M-1:

                Q_prolonged[i*3 + 0] = 0.5*Q_exit[0] + 0.5*Q_mass[-1]
                Q_prolonged[i * 3 + 1] =0.5*Q_exit[1] + 0.5*Q_mom[-1]
                Q_prolonged[i * 3 + 2] = 0.5*Q_exit[2] +0.5*Q_ene[-1]

            else:
                i_coarse_L = i//2 - 1
                i_coarse_R = i//2

                #print("i: ", i, "i_L", i_coarse_L, "i_R: ", i_coarse_R)

                Q_prolonged[i * 3 + 0] = 0.5 * Q_mass[i_coarse_L] + 0.5 * Q_mass[i_coarse_R]
                Q_prolonged[i * 3 + 1] = 0.5 * Q_mom[i_coarse_L] + 0.5 * Q_mom[i_coarse_R]
                Q_prolonged[i * 3 + 2] = 0.5 * Q_ene[i_coarse_L] + 0.5 * Q_ene[i_coarse_R]


        return Q_prolonged

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
