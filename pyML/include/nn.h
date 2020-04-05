#ifndef NN_H
#define NN_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef MPI_ENABLED

	#include "mpi.h"

#endif

enum ActFnEnum{ LINEAR=0, RELU=1, SIGMOID=2, TANH=3 };

typedef struct {

	double **nodes, **d_nodes;
 	double **biases, **d_biases;
 	double ***weights, ***d_weights;
	double *variables, *d_variables_datapoint, *d_variables_batch;
	int nLayers, *nNodes, nVariables, *activations;

} NN;

typedef struct {

	double *m, *v, *mh, *vh;
	double b1, b2, b1t, b2t, a, e;

} AdamOptimizerParams;

void AllocateNN(NN *nn, int nLayers, int *nNodes, int *activations, double *variables);

void DellocateNN(NN *nn);

void AllocateAdam(AdamOptimizerParams *adam, int nVariables, double learning_rate);

void DeallocateAdam(AdamOptimizerParams *adam);

double Activate(int kind, double value);

double dActivate(int kind, double act_value, double d_act_value);

void UpdateVariablesAdam(AdamOptimizerParams *params, int nVariables, double *variables, double *d_variables);

void BackProp(NN *nn);

void ForwProp(NN *nn, double *inputs);

double EvalBatchSens(NN *nn, int nBatchData, int iDataBeg, double *batch_inputs, double *batch_targets, int target_sens_given);

void RunEpoch(NN *nn, AdamOptimizerParams *adam, int nEpochData, double train_frac, int batch_size, double *epoch_inputs, double *epoch_targets);

double Predict(NN *nn, int nBatchData, int iDataBeg, double *batch_inputs, double *batch_targets, double *batch_outputs);

void Print(NN *nn);

void SaveVariables(NN *nn);

void LoadVariables(NN *nn);

void UpdateVariables(NN *nn, double *var_sens, double alpha);

#ifdef MPI_ENABLED

	void Distribute(int nData, int nInputs, int nOutputs, double *inputs, double *outputs);

#endif

#endif
