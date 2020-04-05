#include "../include/nn.h"

void AllocateNN(NN *nn, int nLayers, int *nNodes, int *activations, double *variables) {

	srand(12434);

	int iLayer, iNode, iPrevNode, iVariable;

	nn->nLayers     = nLayers;
	nn->nNodes      = (int*)calloc(nLayers, sizeof(int));
	nn->activations = (int*)calloc(nLayers, sizeof(int));

	for(iLayer=0; iLayer<nLayers; iLayer++) {
	
		nn->nNodes[iLayer]      = nNodes[iLayer];
		nn->activations[iLayer] = activations[iLayer];
	
	}

	nn->nVariables = 0;

	for (iLayer=1; iLayer<nn->nLayers; iLayer++)

		nn->nVariables += nn->nNodes[iLayer] * (nn->nNodes[iLayer-1] + 1);

	int counter = 0;

	nn->variables             = (double*)calloc(nn->nVariables, sizeof(double));
	nn->d_variables_datapoint = (double*)calloc(nn->nVariables, sizeof(double));
	nn->d_variables_batch     = (double*)calloc(nn->nVariables, sizeof(double));

	if (variables != NULL) {
	
		for(iVariable=0; iVariable<nn->nVariables; iVariable++) {
		
			nn->variables[iVariable] = variables[iVariable];
		
		}
	
	}

	else {

		printf("Initializing weights...\n");
	
		for(iVariable=0; iVariable<nn->nVariables; iVariable++) {
		
			nn->variables[iVariable] = 2.0 * (((rand()*1.0) / RAND_MAX) - 0.5) * sqrt(2.0 / nn->nVariables);
		
		}
	
	}

	nn->nodes     =  (double**)calloc(nn->nLayers, sizeof(double*));
	nn->d_nodes   =  (double**)calloc(nn->nLayers, sizeof(double*));

	nn->biases    =  (double**)calloc(nn->nLayers, sizeof(double*));
	nn->d_biases  =  (double**)calloc(nn->nLayers, sizeof(double*));

	nn->weights   = (double***)calloc(nn->nLayers, sizeof(double**));
	nn->d_weights = (double***)calloc(nn->nLayers, sizeof(double**));

	for(iLayer=0; iLayer<nn->nLayers; iLayer++) {
	
		nn->nodes[iLayer]   = (double*)calloc(nn->nNodes[iLayer], sizeof(double));
		nn->d_nodes[iLayer] = (double*)calloc(nn->nNodes[iLayer], sizeof(double));

		if (iLayer > 0) {

			nn->biases[iLayer]   = &(nn->variables[counter]);
			nn->d_biases[iLayer] = &(nn->d_variables_datapoint[counter]);
			
			counter += nn->nNodes[iLayer];
		
			nn->weights[iLayer]   = (double**)calloc(nn->nNodes[iLayer], sizeof(double*));
			nn->d_weights[iLayer] = (double**)calloc(nn->nNodes[iLayer], sizeof(double*));

			for(iNode=0; iNode<nn->nNodes[iLayer]; iNode++) {
			
				nn->weights[iLayer][iNode]   = &(nn->variables[counter]);
				nn->d_weights[iLayer][iNode] = &(nn->d_variables_datapoint[counter]);

				//THIS IS A HACK -------------------------
				//
				//if(iLayer==nn->nLayers-1)
				//	nn->biases[iLayer][iNode] = 0.0;

				counter += nn->nNodes[iLayer-1];
			
			}
		
		}
	
	}

}

void DeallocateNN(NN *nn) {

	int iLayer, iNode, iPrevNode;

	for(iLayer=0; iLayer<nn->nLayers; iLayer++) {
	
		if (iLayer > 0) {

			free(nn->weights[iLayer]);
			free(nn->d_weights[iLayer]);

		}
	
		free(nn->nodes[iLayer]);
		free(nn->d_nodes[iLayer]);

	}

	free(nn->nodes); 
	free(nn->d_nodes);  

	free(nn->biases);
	free(nn->d_biases); 

	free(nn->weights);
	free(nn->d_weights);

	free(nn->variables);
	free(nn->d_variables_datapoint);
	free(nn->d_variables_batch);

	free(nn->nNodes);
	free(nn->activations);

}

void AllocateAdam(AdamOptimizerParams *adam, int nVariables, double learning_rate) {

	adam->m  = (double*)calloc(nVariables, sizeof(double));
	adam->v  = (double*)calloc(nVariables, sizeof(double));
	adam->mh = (double*)calloc(nVariables, sizeof(double));
	adam->vh = (double*)calloc(nVariables, sizeof(double));

	adam->a   = learning_rate;
	adam->e   = 1e-8;
	adam->b1  = 0.9;
	adam->b2  = 0.999;
	adam->b1t = 1.0;
	adam->b2t = 1.0;

}

void DeallocateAdam(AdamOptimizerParams *adam) {

	free(adam->m);
	free(adam->v);
	free(adam->mh);
	free(adam->vh);

}
