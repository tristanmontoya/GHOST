# Tristan Montoya - Euler 1D - Temporal Discretization and Iterative Methods

import numpy as np

class implicitSolver:

    def __init__(self, runName, spatialDiscretization, method, C, isUnsteady= False, max_its = 1000, useLocalTimeStep=False,  ref_u = 300, ref_a = 315, t_f = 1.0, rel_tol = 1.e-12):
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

        #initialization
        self.t = 0.0 #initialize time at 0.0
        self.n = 0 # no time steps taken
        self.Q = spatialDiscretization.u_0_interp
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
                         (self.spatialDiscretization.maxWave(self.Q[i * 3:(i + 1) * 3],
                                                                      self.spatialDiscretization.mesh[i]))
                    self.T[(i*3):(i+1)*3, (i*3):(i+1)*3] = 1./dt*np.eye(3)
            else:
                self.dt = self.C * self.spatialDiscretization.dx / (self.ref_u + self.ref_a)
                self.T = np.eye(self.spatialDiscretization.M*3)*(1./self.dt)

    def runSolver(self):
        if self.method == 0:
            return self.implicitEuler()

    def implicitEuler(self):
        if self.isUnsteady:
            R = self.spatialDiscretization.flowResidual(self.Q)
            for n in range(0,self.n_f):
                dQ = np.linalg.solve(self.implicitEulerLHS(), R)
                self.Q = self.Q + dQ
                R = self.spatialDiscretization.flowResidual(self.Q)
                self.t = self.t + self.dt
                print("step: ", n+1, "time: ", self.t)

        else:
            R = self.spatialDiscretization.flowResidual(self.Q)
            res_0 = np.linalg.norm(R[0::self.spatialDiscretization.n_eq])
            res = res_0
            self.resHistory = [res_0]

            while res/res_0 > self.rel_tol and res > 1.e-10 and self.n < self.max_its:
                dQ = np.linalg.solve(self.implicitEulerLHS(), R)
                self.Q = self.Q + dQ
                R = self.spatialDiscretization.flowResidual(self.Q)
                self.n = self.n + 1
                res = np.linalg.norm(R[0::self.spatialDiscretization.n_eq])
                self.resHistory.append(res)
                print("iteration ", self.n, "res norm = ", res, "rel_res =", res/res_0)

            self.saveResidualHistory()
        self.saveResults()

        return R

    def implicitEulerLHS(self):
        if self.useLocalTimeStep:
            self.setTimeStepMatrix()
        return self.T - self.spatialDiscretization.dRdu(self.Q)

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


        print("pressure in : ", p)
        np.save("../results/"+self.runName+"_results.npy", np.array([Q1, Q2, Q3,
                                                      rho, u, e, p,
                                                       a, Ma, self.spatialDiscretization.mesh]))
