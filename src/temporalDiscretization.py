# Tristan Montoya - Euler 1D - Spatial Discretization

# Takes in a spatialDiscretization object, returns solution

import numpy as np

class TemporalDiscretization:

    def __init__(self, runName, spatialDiscretization, method, C, isUnsteady= False, useLocalTimeStep=False, ref_u = 300, ref_a = 315, t_f = 1.0, rel_tol = 1.e-12):
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

        #initialization
        self.t = 0.0 #initialize time at 0.0
        self.n = 0 # no time steps taken
        self.Q = spatialDiscretization.Q_0

    # return matrix of inverse local time steps
    #this replaces the identity matrix term in the LHS because I divided both sides by the time step (h)
    def timeStepMatrix(self):

        self.T = np.zeros(shape=[3*(self.spatialDiscretization.M), 3*self.spatialDiscretization.M])

        if self.useLocalTimeStep == True:
            # TODO: Local time stepping
            print("local time stepping")
        else:
            for i in range(0, self.spatialDiscretization.M):
                dt = self.C * self.spatialDiscretization.dx / (self.ref_u + self.ref_a)
                self.T[(i*3):(i+1)*3, (i*3):(i+1)*3] = 1./dt*np.eye(3)

        #print("Time step matrix: \n", self.T)

    def implicitEuler(self):
        if self.isUnsteady == False:
            R = self.spatialDiscretization.buildFlowResidual(self.Q)
            res_0 = np.linalg.norm(R)
            res = res_0
            self.resHistory = [res_0]

            while res/res_0 > self.rel_tol and self.n < 1000:
                dQ = np.linalg.solve(self.implicitEulerSystemMatrix(), R)
                self.Q = self.Q + dQ
                R = self.spatialDiscretization.buildFlowResidual(self.Q)
                self.n = self.n + 1
                res = np.linalg.norm(R)
                self.resHistory.append(res)
                print("iteration ", self.n, "res norm = ", res)

            self.saveResidualHistory()
        self.saveResults()

    def implicitEulerSystemMatrix(self):
        return self.T - self.spatialDiscretization.buildFlowJacobian(self.Q)

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
                                                       a, Ma, self.spatialDiscretization.mesh]))
