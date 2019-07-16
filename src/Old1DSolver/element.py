# Tristan Montoya - High Order Spatial Discretization - Elemental Properties

import numpy as np
import quadpy as qp
from scipy import special as sp_special

np.set_printoptions(suppress=True, linewidth=np.nan, threshold=np.nan)
class Element:
    def __init__(self, elementType, p, gridType="uniform", Np=10):
        self.p = p  # operator degree
        if elementType == "dg_dense":
            self.type = 0
            self.Np = p+1
            self.gridType = gridType
            self.generateGrid()
            self.constructDG(diag=False)
        elif elementType == "dg_diag":
            self.type = 1
            self.Np = p+1
            self.gridType = gridType
            self.generateGrid()
            self.constructDG(diag=True)

        elif elementType == "csbp":
            self.type = 2
            #self.Np = 4*p + 1
            self.Np = Np
            self.gridType = "uniform"
            self.generateGrid()
            self.constructCSBP()
            self.V = np.zeros([self.Np, self.Np]) #not using a Legendre Vandermonde for these operators (will look into this at some point)

    def generateGrid(self):

        print("Np: ", self.Np)
        if self.gridType == "lgl":
            quad = qp.line_segment.GaussLobatto(self.Np)
            self.referenceGrid = quad.points
            self.quadratureWeights = quad.weights
        elif self.gridType == "lg":
            quad = qp.line_segment.GaussLegendre(self.Np)
            self.referenceGrid = quad.points
            self.quadratureWeights = quad.weights
        else: #uniform
            self.referenceGrid= np.linspace(-1., 1., self.Np)

        dx = np.zeros(self.Np - 1)
        for i in range(0, self.Np - 1):
            dx[i] = (self.referenceGrid[i + 1] - self.referenceGrid[i])
        self.dx = np.amin(dx)

    def constructCSBP(self):
        dx = 2 / (self.Np - 1)
        num_closures = 2*self.p
        if self.p == 2:

            self.H = dx*np.eye(self.Np)
            self.H[0:num_closures,0:num_closures]=dx*np.array([[0.17e2 / 0.48e2, 0, 0, 0],
                                                               [0, 0.59e2 / 0.48e2, 0, 0],
                                                               [0, 0, 0.43e2 / 0.48e2, 0],
                                                               [0, 0, 0, 0.49e2 / 0.48e2]])

            self.D = np.zeros([self.Np,self.Np])
            self.D[0:num_closures,0:num_closures+self.p] = \
                1/dx*np.array([[-0.24e2 / 0.17e2, 0.59e2 / 0.34e2, -0.4e1 / 0.17e2, -0.3e1 / 0.34e2, 0, 0],
                        [-0.1e1 / 0.2e1, 0, 0.1e1 / 0.2e1, 0, 0, 0],
                        [0.4e1 / 0.43e2, -0.59e2 / 0.86e2, 0, 0.59e2 / 0.86e2, -0.4e1 / 0.43e2, 0],
                        [0.3e1 / 0.98e2, 0, -0.59e2 / 0.98e2, 0, 0.32e2 / 0.49e2, -0.4e1 / 0.49e2]])

            for i in range(num_closures,self.Np-num_closures):
                self.D[i,i-2:i+3]=1/dx*np.array([1/12,-2/3,0,2/3,-1/12])

        if self.p == 3:

            self.H = dx*np.eye(self.Np)
            self.H[0:num_closures,0:num_closures]=dx*np.array([[0.13649e5 / 0.43200e5, 0, 0, 0, 0, 0],
                                                              [0, 0.12013e5 / 0.8640e4, 0, 0, 0, 0],
                                                              [0, 0, 0.2711e4 / 0.4320e4, 0, 0, 0],
                                                              [0, 0, 0, 0.5359e4 / 0.4320e4, 0, 0],
                                                              [0, 0, 0, 0, 0.7877e4 / 0.8640e4, 0],
                                                              [0, 0, 0, 0, 0, 0.43801e5 / 0.43200e5]])


            self.D = np.zeros([self.Np,self.Np])
            self.D[0:num_closures,0:num_closures+self.p] = \
                1/dx*np.array([[-0.21600e5 / 0.13649e5, 0.5124475092222703052468879e25 / 0.2505990045200315292896040e25, -0.58752909548430618941812e23 / 0.313248755650039411612005e24, -0.159267246833799759813661e24 / 0.417665007533385882149340e24, 0.18306575045382041159483e23 / 0.313248755650039411612005e24, 0.120512309461734500607719e24 / 0.2505990045200315292896040e25, 0, 0, 0],
                               [-0.426577465431008328683e21 / 0.918012325152141289800e21, 0, 0.28772124531747103103e20 / 0.91801232515214128980e20, 0.4739194508129391986e19 / 0.22950308128803532245e20, -0.1828286306657606557e19 / 0.61200821676809419320e20, -0.2910908201471785429e19 / 0.114751540644017661225e21, 0, 0, 0],
                               [0.10836021679902364246e20 / 0.114751540644017661225e21, -0.127495216525222408549e21 / 0.183602465030428257960e21, 0, 0.66294394848412989229e20 / 0.91801232515214128980e20, -0.3185918970301186831e19 / 0.22950308128803532245e20, 0.5093573171603569909e19 / 0.306004108384047096600e21, 0, 0, 0],
                               [0.159267246833799759813661e24 / 0.1639876016830108390679400e25, -0.28465971813079192963909e23 / 0.122990701262258129300955e24, -0.179724104434047613799819e24 / 0.491962805049032517203820e24, 0, 0.540747751505546378368499e24 / 0.983925610098065034407640e24, -0.38969683572055185949316e23 / 0.614953506311290646504775e24, 0.72e2 / 0.5359e4, 0, 0],
                               [-0.18306575045382041159483e23 / 0.903897885652927117469325e24, 0.21963203401877827569241e23 / 0.482078872348227795983640e24, 0.17274052656973034997682e23 / 0.180779577130585423493865e24, -0.540747751505546378368499e24 / 0.723118308522341693975460e24, 0, 0.5591070156686698065364559e25 / 0.7231183085223416939754600e25, -0.1296e4 / 0.7877e4, 0.144e3 / 0.7877e4, 0],
                               [-0.120512309461734500607719e24 / 0.8041971570797788126905960e25, 0.34968740224280558358577e23 / 0.1005246446349723515863245e25, -0.13808676868217278023299e23 / 0.1340328595132964687817660e25, 0.77939367144110371898632e23 / 0.1005246446349723515863245e25, -0.5591070156686698065364559e25 / 0.8041971570797788126905960e25, 0, 0.32400e5 / 0.43801e5, -0.6480e4 / 0.43801e5, 0.720e3 / 0.43801e5]])

            for i in range(num_closures,self.Np-num_closures):
                self.D[i,i-3:i+4]=1/dx*np.array([-1/60,3/20,-3/4,0,3/4,-3/20,1/60])


        if self.p == 4:

            self.H = dx * np.eye(self.Np)
            self.H[0:num_closures, 0:num_closures] = dx * np.array([[0.1498139e7 / 0.5080320e7, 0, 0, 0, 0, 0, 0, 0],
                                                                    [0, 0.1107307e7 / 0.725760e6, 0, 0, 0, 0, 0, 0],
                                                                    [0, 0, 0.20761e5 / 0.80640e5, 0, 0, 0, 0, 0],
                                                                    [0, 0, 0, 0.1304999e7 / 0.725760e6, 0, 0, 0, 0],
                                                                    [0, 0, 0, 0, 0.299527e6 / 0.725760e6, 0, 0, 0],
                                                                    [0, 0, 0, 0, 0, 0.103097e6 / 0.80640e5, 0, 0],
                                                                    [0, 0, 0, 0, 0, 0, 0.670091e6 / 0.725760e6, 0],
                                                                    [0, 0, 0, 0, 0, 0, 0, 0.5127739e7 / 0.5080320e7]])

            for i in range(0, num_closures):
                for j in range(0, num_closures):
                    self.H[self.Np - i - 1, self.Np - j - 1] = self.H[i, j]

            self.D = np.zeros([self.Np, self.Np])
            self.D[0:num_closures, 0:num_closures + self.p] = \
                1 / dx * np.array([[-0.16955436044318985087e1, 0.22608643633134057199e1, -0.85367286450337040510e-1, -0.72543619566628612342e0, 0.48823202683514973173e-1, 0.25327067381932511095e0, -0.20810689629510203502e-1, -0.35800463638213927868e-1, 0, 0, 0, 0],
                                   [-0.43697896613650212786e0, 0, 0.11806727914744050659e0, 0.45392741904037194207e0, -0.35451986191651692503e-1, -0.12132305827513695582e0, 0.18087661820535829032e-2, 0.19950546233424744611e-1, 0, 0, 0, 0],
                                   [0.97781066266206924562e-1, -0.69969186172211199969e0, 0, 0.65637393984777096907e0, 0.11267082240058042461e0, -0.33428057204703250861e0, 0.21873662397046339337e0, -0.51590018715877203296e-1, 0, 0, 0, 0],
                                   [0.11897154784237866703e0, -0.38516275383761760282e0, -0.93979240050464527405e-1, 0, 0.14570996944041065112e0, 0.28170129505772535539e0, -0.52460610055305079395e-1, -0.14780208397127463918e-1, 0, 0, 0, 0],
                                   [-0.34885452275029076031e-1, 0.13106074735806541871e0, -0.70285585255172494490e-1, -0.63483881055720004975e0, 0, 0.78722564721145565674e0, -0.23448808805333871951e0, 0.64865185483120361714e-1, -0.86536439119010973969e-2, 0, 0, 0],
                                   [-0.58418504934713186948e-1, 0.14478476223520576322e0, 0.67315236682623567235e-1, -0.39619636345602957631e0, -0.25412458001505128234e0, 0, 0.57501418139033112647e0, -0.10537842745121846328e0, 0.29797181295285022843e-1, -0.27934857464329708915e-2, 0, 0],
                                   [0.66467104043362952119e-2, -0.29889365097445074233e-2, -0.60992789714019610153e-1, 0.10216678579709781702e0, 0.10481488864997796812e0, -0.79622041413657355988e0, 0, 0.82579717441770067391e0, -0.21661535522787203529e0, 0.41260067662451816246e-1, -0.38681313433548577730e-2, 0],
                                   [0.10459594529770367736e-1, -0.30157474178514931388e-1, 0.13159172853630143210e-1, 0.26330727879539228304e-1, -0.26522824364777175297e-1, 0.13347882630163333279e0, -0.75540404471037239532e0, 0, 0.79260196355547737512e0, -0.19815049088886934378e0, 0.37742950645498922625e-1, -0.35384016230155239961e-2]])

            for i in range(num_closures, self.Np - num_closures):
                self.D[i, i - 4:i + 5] = 1 / dx * np.array([1/280, -4/105, 1/5, -4/5, 0, 4/5, -1/5, 4/105, -1/280])


        for i in range(0, num_closures):
            for j in range(0, num_closures):
                self.H[self.Np - i - 1, self.Np - j - 1] = self.H[i, j]
        for i in range(0, num_closures):
            for j in range(0, num_closures + self.p):
                self.D[self.Np - i - 1, self.Np - j - 1] = -self.D[i, j]

        self.Q = self.H @ self.D
        self.t_L = np.zeros(self.Np)
        self.t_L[0] = 1.0
        self.t_R = np.zeros(self.Np)
        self.t_R[self.Np - 1] = 1.0
        self.t_L = self.t_L.reshape(self.Np, 1)
        self.t_R = self.t_R.reshape(self.Np, 1)

    def constructDG(self, diag=True):
        V = np.polynomial.legendre.legvander(self.referenceGrid, self.p)
        modesL = np.zeros(self.Np)
        modesR = np.zeros(self.Np)
        Vr = np.zeros([self.Np, self.Np])
        for j in range(0, self.Np):
            normalizationFactor = np.sqrt(2. / (2 * j + 1))
            V[:, j] /= normalizationFactor  # normalization factor
            modesL[j] = sp_special.legendre(j)(-1.) / normalizationFactor
            modesR[j] = sp_special.legendre(j)(1.) / normalizationFactor
            dPdxi = np.polyder(sp_special.legendre(j))
            Vr[:, j] = dPdxi(self.referenceGrid) / normalizationFactor  # derivatives of legendre basis


        if diag: # collocated quadrature
            self.H = np.diag(self.quadratureWeights)
        else:  # H = (V*V')^-1
            self.H = np.linalg.inv(np.matmul(V, np.transpose(V)))

        # D = Vr*(V)^-1
        self.V = V
        self.D = np.matmul(Vr, np.linalg.inv(V))
        self.Q = self.H @ self.D

        # Projection operators V'*proj(x) = modes(x), reshape to make them column vectors
        self.t_L = np.linalg.solve(np.transpose(V), modesL).reshape(self.Np, 1)
        self.t_R = np.linalg.solve(np.transpose(V), modesR).reshape(self.Np, 1)
