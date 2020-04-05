#include "../include/nn.h"

void PrintNN(NN *nn) {

	int iLayer;

	printf("nLayers:        %d\n", nn->nLayers);

	printf("nNodes:        ");
	for(iLayer=0; iLayer<nn->nLayers; iLayer++)
		printf(" %d", nn->nNodes[iLayer]);
	printf("\n");

	printf("Activations:   ");
	for(iLayer=0; iLayer<nn->nLayers; iLayer++)
		printf(" %d", nn->activations[iLayer]);
	printf("\n");

}

void SaveVariables(NN *nn) {

	int iVariable;

	FILE *fp = fopen("nn_weights.dat", "w");

	for(iVariable=0; iVariable<nn->nVariables; iVariable++)

		fprintf(fp, "%+.15le\n", nn->variables[iVariable]);

	fclose(fp);

}

void LoadVariables(NN *nn) {

	int iVariable, rtnval;

	FILE *fp = fopen("nn_weights.dat", "r");

	for(iVariable=0; iVariable<nn->nVariables; iVariable++)

		rtnval = fscanf(fp, "%le", &(nn->variables[iVariable]));

	fclose(fp);

}
