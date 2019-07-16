# Tristan Montoya - TVD/SSP Runge-Kutta Methods

import numpy as np

class explicitSolver:

    def __init__(self, runName, spatialDiscretization, C, t_f, method='explicit_euler',
                 ref_u = 300, ref_a = 315):
        #problem setup
        self.runName = runName
        self.spatialDiscretization = spatialDiscretization
        self.C = C #Courant number
        self.method=method
        self.t_f = t_f
        self.ref_u = ref_u
        self.ref_a = ref_a

        #initialization
        self.t = 0.0 #initialize time at 0.0
        self.n_f = np.int(np.ceil((self.t_f) / (self.C * self.spatialDiscretization.dx / (self.ref_u + self.ref_a))))
        self.dt = (self.t_f) / self.n_f
        self.n = 0 # no time steps taken
        self.Q = spatialDiscretization.u_0_interp

    def runSolver(self):
        if self.method=='explicit_euler':
            self.explicitEuler()
        if self.method =='SSPRK3':
            self.SSPRK3()

    def explicitEuler(self):
        self.Q, self.isLimited = self.spatialDiscretization.generalizedLimiter(self.Q)
        R = self.spatialDiscretization.flowResidual(self.Q)

        for n in range(0, self.n_f):
            self.Q, self.isLimited = self.spatialDiscretization.generalizedLimiter(self.Q + self.dt*R)

            R = self.spatialDiscretization.flowResidual(self.Q)
            self.t = self.t + self.dt
            print("step: ", n + 1, "time: ", self.t)
        self.saveResults()

    def SSPRK3(self):
        self.Q, self.isLimited = self.spatialDiscretization.generalizedLimiter(self.Q)
        R = self.spatialDiscretization.flowResidual(self.Q)

        for n in range(0, self.n_f):
            Q1, self.isLimited = self.spatialDiscretization.generalizedLimiter(self.Q + self.dt * R)
            R = self.spatialDiscretization.flowResidual(Q1)
            Q2, self.isLimited = self.spatialDiscretization.generalizedLimiter(0.25*(3.0*self.Q + Q1 + self.dt * R))
            R = self.spatialDiscretization.flowResidual(Q2)
            self.Q, self.isLimited = self.spatialDiscretization.generalizedLimiter(1./3.*(self.Q + 2.0*Q2 + 2.0*self.dt*R))
            R = self.spatialDiscretization.flowResidual(self.Q)
            self.t = self.t + self.dt
            print("step: ", n + 1, "time: ", self.t)
        self.saveResults()

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
        np.save("../results/"+self.runName+"_isLimited.npy", self.isLimited)