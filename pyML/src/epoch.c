#include "../include/nn.h"

void RunEpoch(NN *nn, AdamOptimizerParams *adam, int nEpochData, double train_frac, int batch_size, double *epoch_inputs, double *epoch_targets) {

	int iBatch;

	int nBatches = nEpochData / batch_size;
	int nBatchesExcess = nEpochData % batch_size;
	int nBatchesExact  = nBatches - nBatchesExcess;

	int nTrainingData = floor(train_frac*nEpochData);
	int nValidationData = nEpochData - nTrainingData;

	double training_loss, validation_loss;

	training_loss = 0.0;
	validation_loss = 0.0;

	for(iBatch=0; iBatch<nBatchesExcess; iBatch++) {
	
		training_loss += EvalBatchSens(nn, batch_size+1, iBatch*(batch_size+1), epoch_inputs, epoch_targets, 0) / nTrainingData;
		UpdateVariablesAdam(adam, nn->nVariables, nn->variables, nn->d_variables_batch);
	
	}

	for(iBatch=0; iBatch<nBatchesExact; iBatch++) {
	
		training_loss += EvalBatchSens(nn, batch_size, nBatchesExcess*(batch_size+1)+iBatch*batch_size, epoch_inputs, epoch_targets, 0) / nTrainingData;
		UpdateVariablesAdam(adam, nn->nVariables, nn->variables, nn->d_variables_batch);
	
	}

	validation_loss = Predict(nn, nValidationData, nTrainingData, epoch_inputs, epoch_targets, NULL);

	printf("Training_Loss: %.10le   Validation_Loss: %.10le\n", training_loss, validation_loss);

}
