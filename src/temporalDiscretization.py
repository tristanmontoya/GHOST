# Tristan Montoya - Euler 1D - Temporal Discretization and Iterative Methods

import numpy as np

class TemporalDiscretization:

    def __init__(self, runName, spatialDiscretization, method, C, isUnsteady= False, max_its = 1000, useLocalTimeStep=False, useDiagonalForm = False, ref_u = 300, ref_a = 315, t_f = 1.0, rel_tol = 1.e-12):
        #problem setup
        self.runName = runName
        self.spatialDiscretization = spatialDiscretization
        self.C = C #Courant number
        self.method=method
        self.isUnsteady = isUnsteady
        self.useLocalTimeStep = useLocalTimeStep
        self.t_f = t_f
        self.rel_tol = rel_tol
        self.ref_u = ref_u
        self.ref_a = ref_a
        self.max_its = max_its
        self.useDiagonalForm=useDiagonalForm

        #initialization
        self.t = 0.0 #initialize time at 0.0
        self.n = 0 # no time steps taken
        self.Q = spatialDiscretization.Q_0
        self.setTimeStepMatrix()


    #update matrix of inverse local time steps
    #this replaces the identity matrix term in the LHS because we have divided both sides by the time step
    def setTimeStepMatrix(self):

        self.T = np.zeros(shape=[3*(self.spatialDiscretization.M), 3*self.spatialDiscretization.M])

        if self.isUnsteady:
            self.n_f = np.int(np.ceil((self.t_f) / (self.C * self.spatialDiscretization.dx / (self.ref_u + self.ref_a))))
            self.dt = (self.t_f) / self.n_f
            self.T = np.eye(self.spatialDiscretization.M*3)*(1./self.dt)
        else:
            if self.useLocalTimeStep == True:
                for i in range(0, self.spatialDiscretization.M):

                    dt = self.C * self.spatialDiscretization.dx / \
                         (self.spatialDiscretization.problem.specrA_j(self.Q[i * 3:(i + 1) * 3],
                                                                      self.spatialDiscretization.mesh[i]))
                    self.T[(i*3):(i+1)*3, (i*3):(i+1)*3] = 1./dt*np.eye(3)
            else:
                self.dt = self.C * self.spatialDiscretization.dx / (self.ref_u + self.ref_a)
                self.T = np.eye(self.spatialDiscretization.M*3)*(1./self.dt)

    def runSolver(self):
        if self.method == 0:
            self.implicitEuler()
        if self.method == 1:
            self.secondBackwards()

    def implicitEuler(self):
        if self.isUnsteady:
            R = self.implicitEulerRHS()
            for n in range(0,self.n_f):
                X = np.linalg.solve(self.implicitEulerLHS(), R)

                if self.useDiagonalForm:
                    dQ = np.zeros(3*self.spatialDiscretization.M)
                    for j in range(0, self.spatialDiscretization.M):
                        dQ[j*3:(j+1)*3] = self.spatialDiscretization.T_j[j, :, :] @ X[j*3:(j+1)*3]
                else:
                    dQ = X

                self.Q = self.Q + dQ
                R = self.implicitEulerRHS()
                self.t = self.t + self.dt
                print("step: ", n+1, "time: ", self.t)

        else:
            R = self.implicitEulerRHS()
            res_0 = np.linalg.norm(R)
            res = res_0
            self.resHistory = [res_0]

            while res/res_0 > self.rel_tol and self.n < self.max_its:
                X = np.linalg.solve(self.implicitEulerLHS(), R)

                if self.useDiagonalForm:
                    dQ = np.zeros(3*self.spatialDiscretization.M)
                    for j in range(0, self.spatialDiscretization.M):
                        dQ[j*3:(j+1)*3] = self.spatialDiscretization.T_j[j, :, :] @ X[j*3:(j+1)*3]
                else:
                    dQ = X

                self.Q = self.Q + dQ
                R = self.implicitEulerRHS()
                self.n = self.n + 1
                res = np.linalg.norm(R)
                self.resHistory.append(res)
                print("iteration ", self.n, "res norm = ", res)

            self.saveResidualHistory()
        self.saveResults()

    def secondBackwards(self):
        if self.isUnsteady:

            #first step implicit Euler
            R = self.implicitEulerRHS()
            X = np.linalg.solve(self.implicitEulerLHS(), R)
            if self.useDiagonalForm:
                dQ = np.zeros(3 * self.spatialDiscretization.M)
                for j in range(0, self.spatialDiscretization.M):
                    dQ[j * 3:(j + 1) * 3] = self.spatialDiscretization.T_j[j, :, :] @ X[j * 3:(j + 1) * 3]
            else:
                dQ = X

            self.dQ_previous = dQ
            self.Q = self.Q + dQ
            R = self.secondBackwardsRHS()
            self.t = self.t + self.dt
            print("step: ", 1, "time: ", self.t)

            #second-order backwards
            for n in range(1, self.n_f):
                X = np.linalg.solve(self.secondBackwardsLHS(), R)

                if self.useDiagonalForm:
                    dQ = np.zeros(3 * self.spatialDiscretization.M)
                    for j in range(0, self.spatialDiscretization.M):
                        dQ[j * 3:(j + 1) * 3] = self.spatialDiscretization.T_j[j, :, :] @ X[j * 3:(j + 1) * 3]
                else:
                    dQ = X

                self.dQ_previous = dQ
                self.Q = self.Q + dQ
                R = self.secondBackwardsRHS()
                self.t = self.t + self.dt
                print("step: ", n + 1, "time: ", self.t)

        else:
            R = self.implicitEulerRHS()
            res_0 = np.linalg.norm(R)
            self.resHistory = [res_0]

            print("For steady problems, use implicit Euler.")

            self.saveResidualHistory()
        self.saveResults()

    def implicitEulerLHS(self):
        if self.useLocalTimeStep:
            self.setTimeStepMatrix()
        if self.useDiagonalForm:
            return self.T - self.spatialDiscretization.buildFlowJacobianDiagonalForm(self.Q)
        else:
            return self.T - self.spatialDiscretization.buildFlowJacobian(self.Q)

    def secondBackwardsLHS(self):
        if self.useLocalTimeStep:
            self.setTimeStepMatrix()
        if self.useDiagonalForm:
            return self.T - 2./3.*self.spatialDiscretization.buildFlowJacobianDiagonalForm(self.Q)
        else:
            return self.T - 2./3.*self.spatialDiscretization.buildFlowJacobian(self.Q)

    def implicitEulerRHS(self):
        if self.useDiagonalForm:
            return self.spatialDiscretization.buildFlowResidualDiagonalForm(self.Q)
        else:
            return self.spatialDiscretization.buildFlowResidual(self.Q)

    def secondBackwardsRHS(self):
        if self.useDiagonalForm:
            X_previous = np.zeros(3*self.spatialDiscretization.M)

            # Need to have entire RHS multiplied by T^-1
            for j in range(0, self.spatialDiscretization.M):
                X_previous[j * 3:(j + 1) * 3] = self.spatialDiscretization.T_j_inv[j, :, :] @ self.dQ_previous[j * 3:(j + 1) * 3]
            return 1./3.*self.T @ X_previous + 2./3.*self.spatialDiscretization.buildFlowResidualDiagonalForm(self.Q)
        else:
            return 1./3.*self.T @ self.dQ_previous + 2./3.*self.spatialDiscretization.buildFlowResidual(self.Q)

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
