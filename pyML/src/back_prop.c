#include "../include/nn.h"

void BackProp(NN *nn) {

	int iLayer, iNode, iPrevNode;

	for (iLayer=nn->nLayers-1; iLayer>=1; iLayer--) {

		for (iPrevNode=0; iPrevNode<nn->nNodes[iLayer-1]; iPrevNode++) {
		
			nn->d_nodes[iLayer-1][iPrevNode] = 0.0;
		
		}
	
		for (iNode=0; iNode<nn->nNodes[iLayer]; iNode++) {
		
			nn->d_nodes[iLayer][iNode] = dActivate(nn->activations[iLayer], nn->nodes[iLayer][iNode], nn->d_nodes[iLayer][iNode]);
		
			for (iPrevNode=0; iPrevNode<nn->nNodes[iLayer-1]; iPrevNode++) {

				nn->d_weights[iLayer][iNode][iPrevNode]  = nn->d_nodes[iLayer][iNode] * nn->nodes[iLayer-1][iPrevNode];
				nn->d_nodes[iLayer-1][iPrevNode] += nn->d_nodes[iLayer][iNode] * nn->weights[iLayer][iNode][iPrevNode];
			
			}

			nn->d_biases[iLayer][iNode] = nn->d_nodes[iLayer][iNode];

			// THIS IS JUST A HACK ------------------------------------
			
			//if (iLayer==nn->nLayers-1)
			//	nn->d_biases[iLayer][iNode] = 0.0;

		}
	
	}

}
