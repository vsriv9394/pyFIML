import numpy as np
from RHT import RHT
from pyML.NN import NN

def FIML_Classic(nOptimIter=20, step=0.01, cases=[], data=[]):

    for iOptimIter in range(nOptimIter):

        obj_total = 0.0

        for case, case_data in zip(cases, data):

            case.direct_solve()
            obj, beta_sens = case.adjoint_solve(case_data)
            obj_total += obj
            case.beta -= beta_sens / np.max(np.abs(beta_sens)) * step

        print("Iteration %6d    Objective Function %.10le"%(iOptimIter, obj_total))

    features = []
    beta     = []

    for case in cases:

        case.direct_solve()
        features.append(case.getFeatures())
        beta.append(case.getBeta())

    features = np.hstack(features)
    beta     = np.hstack(beta)

    nn = NN({"shape":[ features.shape[0] , 7 , 7 , beta.shape[0] ], "actfn":[0,2,2,0], "vars":None})

    nn.Train(features, beta, nEpochs=2000)

    for case in cases:

        case.nn = nn
        case.plot = True
        case.direct_solve()

if __name__=="__main__":

    FIML_Classic(nOptimIter=1000,
                step=0.01,
                cases=[RHT(T_inf=50)],
                data=[np.loadtxt("True_solutions/solution_50")])
