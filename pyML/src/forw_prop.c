#include "../include/nn.h"

void ForwProp(NN *nn, double *inputs) {

	int iInput, iLayer, iNode, iPrevNode;

	for (iInput=0; iInput<nn->nNodes[0]; iInput++) {

		nn->nodes[0][iInput] = inputs[iInput];

	}

	for (iLayer=1; iLayer<nn->nLayers; iLayer++) {
	
		for (iNode=0; iNode<nn->nNodes[iLayer]; iNode++) {
		
			nn->nodes[iLayer][iNode] = nn->biases[iLayer][iNode];

			for (iPrevNode=0; iPrevNode<nn->nNodes[iLayer-1]; iPrevNode++) {
			
				nn->nodes[iLayer][iNode] += nn->weights[iLayer][iNode][iPrevNode] * nn->nodes[iLayer-1][iPrevNode];
			
			}

			nn->nodes[iLayer][iNode] = Activate(nn->activations[iLayer], nn->nodes[iLayer][iNode]);
		
		}
	
	}

}
