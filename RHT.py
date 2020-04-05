import numpy as np
import matplotlib.pyplot as plt
from pyML.NN import NN

def setJacobian(jacT, jacbeta, T, T_inf, beta, dy2):

    jacT[0,0]   = -1.0
    jacT[-1,-1] = -1.0

    i = np.linspace(1,jacT.shape[0]-2,jacT.shape[0]-2).astype(int)

    jacT[i,i-1] = 1. / dy2
    jacT[i,i]   = - 2. / dy2 - 2E-3 * beta[i] * T[i]**3
    jacT[i,i+1] = 1. / dy2

    jacbeta[i,i] = 5E-4 * (T_inf**4 - T[i]**4)

class RHT:

    # Class to generate data from the Radiative heat transfer model

    def __init__(self, T_inf=5.0, npoints=129, dt=1e-2, n_iter=1000, tol=1e-8, nn=None, verbose=False, plot=False):

        self.T_inf      = T_inf                             # Temperature of the one-dimensional body
        self.y          = np.linspace(0., 1., npoints)      # Coordinates
        self.dy2        = (self.y[1]-self.y[0])**2          # dy^2 to be used in the second derivative
        self.T          = np.zeros_like(self.y)             # Initial temperature of the body at the coordinates specified above
        self.beta       = np.ones_like(self.y)              # Augmentation profiles of the model
        self.dt         = dt                                # Time step to be used in the simulation
        self.n_iter     = n_iter                            # Maximum number of iterations to be run during direct solve
        self.tol        = tol                               # Maximum value of residual at which direct solve can be terminated
        self.res        = np.zeros_like(self.y)             # Residuals
        self.jacT       = np.zeros((npoints, npoints))      # dRdT
        self.jacbeta    = np.zeros((npoints, npoints))      # dRdbeta
        self.nn         = nn                                # Augmentation Neural Network
        self.verbose    = verbose
        self.plot       = plot

    #-----------------------------------------------------------------------------------------------------------------------------------

    def getFeatures(self):

        features = np.zeros((2, self.y.size))
        features[0,:] = self.T / self.T_inf
        features[1,:] = self.y
        return features

    #-----------------------------------------------------------------------------------------------------------------------------------

    def getBeta(self):

        return self.beta.reshape((1, self.y.size))

    #-----------------------------------------------------------------------------------------------------------------------------------

    def evalResidual(self):
        
        self.res[1:-1] = (self.T[0:-2]-2*self.T[1:-1]+self.T[2:])/self.dy2 + 5E-4*self.beta[1:-1]*(self.T_inf**4-self.T[1:-1]**4)
        self.res[0]  = -self.T[0]
        self.res[-1] = -self.T[-1]
        
    #-----------------------------------------------------------------------------------------------------------------------------------
    
    def implicitEulerUpdate(self):

        # Update the states using implicit Euler time stepping

        if self.nn!=None:
            features = self.getFeatures()
            self.beta = self.nn.Predict(features)[0,:]
        
        setJacobian(self.jacT, self.jacbeta, self.T, self.T_inf, self.beta, self.dy2)
        self.evalResidual()

        self.T = self.T + np.linalg.solve(np.eye(np.shape(self.y)[0])/self.dt - self.jacT, self.res)

        return np.linalg.norm(self.res)

    #-----------------------------------------------------------------------------------------------------------------------------------

    def direct_solve(self):

        # Iteratively solve the equations for the model until either the tolerance is achieved or the maximum iterations have been done

        for iteration in range(self.n_iter):

            # Update the states for this iteration

            res_norm = self.implicitEulerUpdate()
            if self.verbose:
                print("%9d\t%E"%(iteration, res_norm))


            # Check if the residual is within tolerance, if yes, save the data and exit the simulation, if no, continue
            
            if res_norm<self.tol:
                break
        

        # Once the simulation is terminated, show the results if plot is True

        if self.plot:

            plt.plot(self.y, self.T, '-r', LineWidth=2.0, label='Solver')
            plt.plot(self.y, np.loadtxt('True_solutions/solution_%d'%(int(self.T_inf))), '-k', LineWidth=2.0, label='Exact')
            plt.xlabel("x")
            plt.ylabel("T")
            plt.legend()
            plt.tight_layout(pad=1.01)
            plt.show()

            plt.plot(self.y, self.beta)
            plt.xlabel("x")
            plt.ylabel("beta")
            plt.tight_layout(pad=1.01)
            plt.show()

    #-----------------------------------------------------------------------------------------------------------------------------------

    def adjoint_solve(self, data, reg=1e-4):
        
        # Solve the discrete adjoint equation to obtain sensitivities of the objective function w.r.t. augmentation field

        psi  = np.linalg.solve(self.jacT.T, 2.0 * (self.T - data) / self.T.size)
        sens = 2.0 * reg * (self.beta - 1) / self.beta.size - np.matmul(self.jacbeta.T, psi)

        obj = np.sum((self.T - data)**2) / self.T.size;

        return obj, sens



if __name__=="__main__":

    for T_inf in np.linspace(5.,50.,10):
        
        rht = RHT(T_inf=T_inf)
        rht.direct_solve()
