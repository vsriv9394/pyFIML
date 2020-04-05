import numpy as np
from RHT import RHT
from pyML.NN import NN

def FIML_Direct(nOptimIter=20, step=0.01, cases=[], data=[]):

    features = []
    beta     = []

    for case in cases:

        case.direct_solve()
        features.append(case.getFeatures())
        beta.append(case.getBeta())

    features = np.hstack(features)
    beta     = np.hstack(beta)

    print (beta.shape)

    nn = NN({"shape":[ features.shape[0] , 7 , 7 , beta.shape[0] ], "actfn":[0,2,2,0], "vars":None})

    nn.Train(features, beta, nEpochs=2000)

    for iOptimIter in range(nOptimIter):

        nn_sens   = 0.0
        obj_total = 0.0

        for case, case_data in zip(cases, data):
            
            case.nn = nn

            case.direct_solve()
            obj, beta_sens = case.adjoint_solve(case_data)

            nn_sens += nn.GetSens(features, beta_sens)
            obj_total += obj

        print("Iteration %6d    Objective Function %.10le"%(iOptimIter, obj_total))
            
        nn.UpdateVariables(nn_sens/len(cases), step)

    for case in cases:

        case.nn = nn
        case.plot = True
        case.direct_solve()

if __name__=="__main__":

    FIML_Direct(nOptimIter=500,
                step=0.02,
                cases=[RHT(T_inf=40)],
                data=[np.loadtxt("True_solutions/solution_40")])
