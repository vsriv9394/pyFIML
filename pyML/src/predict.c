#include "../include/nn.h"

double Predict(NN *nn, int nBatchData, int iDataBeg, double *batch_inputs, double *batch_targets, double *batch_outputs) {

	double batch_loss = 0.0;
	double datapoint_loss;

	int iData, iOutput;
	int nInputs = nn->nNodes[0];
	int nOutputs = nn->nNodes[nn->nLayers-1];

	for(iData=0; iData<nBatchData; iData++) {

		datapoint_loss = 0.0;
	
		ForwProp(nn, &(batch_inputs[(iDataBeg+iData)*nInputs]));

		for (iOutput=0; iOutput<nOutputs; iOutput++) {

			if (batch_targets != NULL)
				datapoint_loss += pow(nn->nodes[nn->nLayers-1][iOutput] - batch_targets[(iDataBeg+iData)*nOutputs+iOutput], 2);
			if (batch_outputs != NULL)
				batch_outputs[(iDataBeg+iData)*nOutputs+iOutput] = nn->nodes[nn->nLayers-1][iOutput];

		}

		batch_loss += datapoint_loss / nOutputs;

	}

	return batch_loss / nBatchData;

}
