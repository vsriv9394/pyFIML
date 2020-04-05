import sys
import os
import ctypes as C
import numpy as np
import matplotlib.pyplot as plt

CPtr = C.POINTER

lib = C.CDLL(os.path.dirname(os.path.abspath(__file__)) + '/lib/nn.so')

class AdamOptimizerParams(C.Structure):

    _fields_ = [('m',    CPtr(C.c_double)),
                ('v',    CPtr(C.c_double)),
                ('mh',   CPtr(C.c_double)),
                ('vh',   CPtr(C.c_double)),
                ('b1',   C.c_double),
                ('b2',   C.c_double),
                ('b1t',  C.c_double),
                ('b2t',  C.c_double),
                ('a',    C.c_double),
                ('e',    C.c_double)]

    def __init__(self):
        pass




class NN(C.Structure):

    _fields_ = [('nodes',            CPtr(CPtr(C.c_double))),
                ('d_nodes',          CPtr(CPtr(C.c_double))),
                ('biases',           CPtr(CPtr(C.c_double))),
                ('d_biases',         CPtr(CPtr(C.c_double))),
                ('weights',     CPtr(CPtr(CPtr(C.c_double)))),
                ('d_weights',   CPtr(CPtr(CPtr(C.c_double)))),
                ('variables',             CPtr(C.c_double)),
                ('d_variables_datapoint', CPtr(C.c_double)),
                ('d_variables_batch',     CPtr(C.c_double)),
                ('nLayers',                    C.c_int  ),
                ('nNodes',                CPtr(C.c_int  )),
                ('nVariables',                 C.c_int  ),
                ('activations',           CPtr(C.c_int  ))]

    def __init__(self, nn_dict, learning_rate=0.001):

        self.adam = AdamOptimizerParams()

        self.Allocate          = lib.__getattr__('AllocateNN')
        self.Allocate.restype  = None
        self.Allocate.argtypes = [CPtr(NN),
                                  C.c_int,
                                  CPtr(C.c_int),
                                  CPtr(C.c_int),
                                  CPtr(C.c_double)]

        self.AllocateAdam          = lib.__getattr__('AllocateAdam')
        self.AllocateAdam.restype  = None
        self.AllocateAdam.argtypes = [CPtr(AdamOptimizerParams),
                                      C.c_int,
                                      C.c_double]

        self.DeallocateAdam          = lib.__getattr__('DeallocateAdam')
        self.DeallocateAdam.restype  = None
        self.DeallocateAdam.argtypes = [CPtr(AdamOptimizerParams)]

        self.Deallocate          = lib.__getattr__('DeallocateNN')
        self.Deallocate.restype  = None
        self.Deallocate.argtypes = [CPtr(NN)]

        self.printfn          = lib.__getattr__('PrintNN')
        self.printfn.restype  = None
        self.printfn.argtypes = [CPtr(NN)]

        self.savefn          = lib.__getattr__('SaveVariables')
        self.savefn.restype  = None
        self.savefn.argtypes = [CPtr(NN)]

        self.loadfn          = lib.__getattr__('LoadVariables')
        self.loadfn.restype  = None
        self.loadfn.argtypes = [CPtr(NN)]

        self.EvalBatchSens          = lib.__getattr__('EvalBatchSens')
        self.EvalBatchSens.restype  = C.c_double
        self.EvalBatchSens.argtypes = [CPtr(NN),
                                       C.c_int,
                                       C.c_int,
                                       CPtr(C.c_double),
                                       CPtr(C.c_double),
                                       C.c_int]

        self.RunEpoch          = lib.__getattr__('RunEpoch')
        self.RunEpoch.restype  = None
        self.RunEpoch.argtypes = [CPtr(NN),
                                  CPtr(AdamOptimizerParams),
                                  C.c_int,
                                  C.c_double,
                                  C.c_int,
                                  CPtr(C.c_double),
                                  CPtr(C.c_double)]

        self.predict          = lib.__getattr__('Predict')
        self.predict.restype  = C.c_double
        self.predict.argtypes = [CPtr(NN),
                                 C.c_int,
                                 C.c_int,
                                 CPtr(C.c_double),
                                 CPtr(C.c_double),
                                 CPtr(C.c_double)]

        self.UpdateVars          = lib.__getattr__('UpdateVariables')
        self.UpdateVars.restype  = None
        self.UpdateVars.argtypes = [CPtr(NN), CPtr(C.c_double), C.c_double]

        shape_arr = np.array(nn_dict["shape"], dtype=np.int32)
        actfn_arr = np.array(nn_dict["actfn"], dtype=np.int32)

        if nn_dict["vars"]!=None:
            
            self.Allocate(self,
                          C.c_int(len(nn_dict["shape"])),
                          shape_arr.ctypes.data_as(CPtr(C.c_int)),
                          actfn_arr.ctypes.data_as(CPtr(C.c_int)),
                          nn_dict["vars"].ctypes.data_as(CPtr(C.c_double)))
        
        else:

            self.Allocate(self,
                          C.c_int(len(nn_dict["shape"])),
                          shape_arr.ctypes.data_as(CPtr(C.c_int)),
                          actfn_arr.ctypes.data_as(CPtr(C.c_int)),
                          None)

        self.AllocateAdam(self.adam, self.nVariables, C.c_double(learning_rate))

    
    
    
    def __del__(self):

        self.Deallocate(self)
        self.DeallocateAdam(self.adam)

    
    
    
    def Save(self):

        self.savefn(self)



    
    def Load(self):

        self.loadfn(self)



    
    def Print(self):

        self.printfn(self)



    
    def UpdateVariables(self, var_sens, alpha):

        self.UpdateVars(self, var_sens.ctypes.data_as(CPtr(C.c_double)), alpha)

    
    
    
    def GetSens(self, inputs, output_sens):

        self.EvalBatchSens(self,
                           C.c_int(inputs.shape[1]),
                           C.c_int(0),
                           inputs.ctypes.data_as(CPtr(C.c_double)),
                           output_sens.ctypes.data_as(CPtr(C.c_double)),
                           C.c_int(1))

        return np.ctypeslib.as_array((C.c_double * self.nVariables).from_address(C.addressof(self.d_variables_batch.contents))).copy()

    
    
    
    def Train(self, inputs, targets, nEpochs=100, training_fraction=0.85, batch_size=32):

        ind = np.arange(inputs.shape[1])
        np.random.shuffle(ind)

        s_inputs  = inputs[:,ind]
        s_targets = targets[:,ind]

        for iEpoch in range(nEpochs):

            sys.stdout.write("Iteration %6d    "%(iEpoch+1))
            sys.stdout.flush()

            self.RunEpoch(self,
                          self.adam,
                          C.c_int(inputs.shape[1]),
                          C.c_double(training_fraction),
                          C.c_int(batch_size),
                          s_inputs.ctypes.data_as(CPtr(C.c_double)),
                          s_targets.ctypes.data_as(CPtr(C.c_double)))

            

    
    def Predict(self, inputs):

        outputs = np.zeros((self.nNodes[self.nLayers-1], inputs.shape[1]))

        ind = np.arange(inputs.shape[1])
        s_inputs = inputs[:,ind]

        self.predict(self,
                     C.c_int(inputs.shape[1]),
                     C.c_int(0),
                     s_inputs.ctypes.data_as(CPtr(C.c_double)),
                     None,
                     outputs.ctypes.data_as(CPtr(C.c_double)))

        return outputs




if __name__=="__main__":
    
    nn = NN({"shape":[1,7,7,1], "actfn":[0,2,2,0], "vars":None}, learning_rate=0.01)
    nn.Print()

    x = np.zeros((1,257))

    x[0,:] = np.linspace(0.,1.,257)
    #x[1,:] = np.linspace(2.,3.,201)
    #y = x[0,:]**0.5 * x[1,:]**2
    #y = y.reshape((1,201))
    y = x * (x - 1./3.) * (x - 2./3.)

    #nn.Train(x, y, nEpochs=50000, training_fraction=0.8, batch_size=16)
    '''
    for i in range(100000):
        yp = nn.Predict(x)
        sens = nn.GetSens(x, 2*(yp-y))
        print("%9d %.10le"%(i, np.sum((yp-y)**2)))
        nn.UpdateVariables(sens/np.max(np.abs(sens)), 1e-3)
    '''

    nn.Load()

    yp = nn.Predict(x)

    #nn.Save()

    plt.plot(x[0,:],y[0,:])
    plt.plot(x[0,:],yp[0,:])
    plt.show()
