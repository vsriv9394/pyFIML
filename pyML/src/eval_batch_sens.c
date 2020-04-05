#include "../include/nn.h"

double EvalBatchSens(NN *nn, int nBatchData, int iDataBeg, double *batch_inputs, double *batch_targets, int target_sens_given) {

	double batch_loss = 0.0;
	double datapoint_loss;

	int iData, iOutput, iVariable;
	int nInputs = nn->nNodes[0];
	int nOutputs = nn->nNodes[nn->nLayers-1];

	for(iVariable=0; iVariable<nn->nVariables; iVariable++)

		nn->d_variables_batch[iVariable] = 0.0;

	for(iData=0; iData<nBatchData; iData++) {

		datapoint_loss = 0.0;

		ForwProp(nn, &(batch_inputs[(iDataBeg+iData)*nInputs]));

		if (target_sens_given==0) {

			for (iOutput=0; iOutput<nOutputs; iOutput++) {

				datapoint_loss += pow(nn->nodes[nn->nLayers-1][iOutput] - batch_targets[(iDataBeg+iData)*nOutputs+iOutput], 2);
				nn->d_nodes[nn->nLayers-1][iOutput] = 2.0 * (nn->nodes[nn->nLayers-1][iOutput] - batch_targets[(iDataBeg+iData)*nOutputs+iOutput]);

			}

			batch_loss += datapoint_loss / nOutputs;

		}

		else {
		
			for (iOutput=0; iOutput<nOutputs; iOutput++) {
			
				nn->d_nodes[nn->nLayers-1][iOutput] = batch_targets[(iDataBeg+iData)*nOutputs+iOutput];
			
			}
		
		}

		BackProp(nn);

		for(iVariable=0; iVariable<nn->nVariables; iVariable++) {
		
			nn->d_variables_batch[iVariable] += nn->d_variables_datapoint[iVariable];
		
		}

	}

	#ifdef MPI_ENABLED

	int nBatchDataGlobal;
	MPI_Allreduce(&nBatchData, &nBatchDataGlobal, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	nBatchData = nBatchDataGlobal;

	double *d_variables_batch_global = (double*)calloc(nn->nVariables, sizeof(double));
	MPI_Allreduce(&d_variables_batch, &d_variables_batch_global, nn->nVariables, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	for(iVariable=0; iVariable<nn->nVariables; iVariable++)
		d_variables_batch[iVariable] = d_variables_batch_global[iVariable];
	free(d_variables_batch_global);

	#endif
	
	for(iVariable=0; iVariable<nn->nVariables; iVariable++) {
	
		nn->d_variables_batch[iVariable] = nn->d_variables_batch[iVariable] / nBatchData;
	
	}

	return batch_loss / nBatchData;

}
