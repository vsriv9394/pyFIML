#include "../include/nn.h"

void UpdateVariables(NN *nn, double *var_sens, double alpha) {

	int iVariable, iLayer, iNode;

	for(iVariable=0; iVariable<nn->nVariables; iVariable++) {
	
		nn->variables[iVariable] -= var_sens[iVariable] * alpha;
	
	}

	// THIS IS JUST A HACK ------------------------------------

	//for(iNode=0; iNode<nn->nNodes[nn->nLayers-1]; iNode++)
	//	nn->biases[nn->nLayers-1][iNode] = 0.0;

}

void UpdateVariablesAdam(AdamOptimizerParams *params, int nVariables, double *variables, double *d_variables) {

	int iVariable;

	params->b1t *= params->b1;
	params->b2t *= params->b2;

	for(iVariable=0; iVariable<nVariables; iVariable++) {
	
		params->m[iVariable] *= params->b1;
		params->m[iVariable] += d_variables[iVariable] * (1.0 - params->b1);
	
		params->v[iVariable] *= params->b2;
		params->v[iVariable] += pow(d_variables[iVariable], 2) * (1.0 - params->b2);

		params->mh[iVariable] = params->m[iVariable] / (1.0 - params->b1t);
		params->vh[iVariable] = params->v[iVariable] / (1.0 - params->b2t);

		variables[iVariable] -= params->a * params->mh[iVariable] / (sqrt(params->vh[iVariable]) + params->e);
	
	}

}
