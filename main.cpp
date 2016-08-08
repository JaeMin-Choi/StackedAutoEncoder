#include <iostream>
#include <fstream>
#include "Layer.h"
#include "SimpleAutoEncoder.h"
#include "StackedAutoEncoder.h"
using namespace std;

#define MAX_EPOCH 1000000
#define MAX_ERROR 0.001f

#define INPUT_DATA_DIMENSION 100
#define TRAINING_DATA_SIZE 4

#define LEARNING_RATE 0.05f



int main(int argc, char* argv[]) {
	FILE* myfile = fopen("result.txt", "a");
	float training_data[TRAINING_DATA_SIZE][INPUT_DATA_DIMENSION] = {
		{ 1,1,0,0,0,0,1,1,0,0,
		0,1,0,0,0,0,0,0,1,0,
		0,0,1,0,0,1,0,0,0,0,
		0,1,0,0,1,1,1,0,1,0,
		0,1,0,0,1,0,0,1,0,0,
		0,1,0,0,0,0,0,0,0,0,
		0,1,0,0,0,1,0,1,0,1,
		0,0,0,0,0,1,0,0,0,0,
		0,0,0,0,0,0,1,0,0,0,
		0,0,0,0,0,0,1,1,1,1 },

		{ 1,1,0,0,0,0,1,1,0,0,
		0,0,0,1,0,0,0,1,1,0,
		0,0,1,0,0,1,0,0,0,0,
		0,1,0,0,1,0,1,1,1,0,
		1,0,0,0,1,0,0,1,0,0,
		0,1,0,0,0,1,0,0,0,0,
		1,1,0,0,0,1,0,1,0,1,
		0,0,0,0,0,0,0,1,0,0,
		0,1,0,0,0,0,1,0,0,0,
		0,1,0,0,0,0,0,1,1,1 },

		{ 1,0,0,1,0,0,1,1,0,0,
		0,1,0,0,0,0,0,0,1,0,
		0,0,1,0,0,1,0,0,1,0,
		0,1,0,0,1,1,1,0,1,0,
		1,0,0,0,1,0,0,1,0,0,
		0,1,0,0,0,1,1,0,1,0,
		0,1,0,1,0,0,0,1,0,1,
		0,0,0,0,0,1,0,0,0,0,
		0,1,0,1,0,0,1,0,0,0,
		0,0,0,0,0,0,1,1,1,1 },

		{ 0,1,0,0,1,0,1,0,1,0,
		0,0,0,1,1,0,0,1,1,0,
		0,0,1,0,0,1,0,0,0,0,
		1,1,0,0,1,0,1,1,1,0,
		1,0,1,0,1,0,0,1,0,0,
		0,1,0,0,0,1,0,0,0,0,
		1,1,1,0,0,1,0,1,0,1,
		0,1,0,0,0,0,1,1,0,0,
		0,0,1,0,1,0,0,0,1,0,
		0,1,0,1,1,0,0,1,0,1 },

	};
	int num_AE = 3;
	int eachEncoder_outDim[] = { 50,25,10 };

	StackedAutoEncoder myStackedAE(num_AE, INPUT_DATA_DIMENSION, eachEncoder_outDim);


	printf("Start Training Stacked-AutoEncoder \n");
	fprintf(myfile, "Start Training Stacked-AutoEncoder \n");

	
	for (int idx = 0; idx < num_AE; idx++) {
		int for_breaking = 1;
		for (int epoch = 0; epoch < MAX_EPOCH && for_breaking == 1; epoch++) {
			float error = 0.f;
			for (int i = 0; i < TRAINING_DATA_SIZE; i++) {
				myStackedAE.Back_Propagate(training_data[i], idx);
				error += myStackedAE.Get_Reproduct_Error(idx);
			}

			error /= TRAINING_DATA_SIZE;
			if (error < MAX_ERROR) {
				for_breaking = 0;
				break;
			}
			myStackedAE.Weight_Update(LEARNING_RATE, idx);



			if ((epoch + 1) % 100 == 0) {
				printf("%d-th autoencoder, epoch = %d, error = %f \n", idx + 1, epoch + 1, error);
				fprintf(myfile, "%d-th autoencoder, epoch = %d, error = %f \n", idx + 1, epoch + 1, error);
			}
		}
	}



	fprintf(myfile, "Finish Training Stacked-AutoEncoder \n\n");


	printf("Testing Stacked-AutoEncoder\n");
	fprintf(myfile, "Testing Stacked-AutoEncoder\n");

	for (int number = 0; number < num_AE; number++) {
		fprintf(myfile, "\n\n%d-th autoencoder testing \n", number + 1);
		for (int j = 0; j < TRAINING_DATA_SIZE; j++) {
			myStackedAE.Decoding(training_data[j], number);
			float *encoded = myStackedAE.Get_Encoding_Result(number);
			float *reprod = myStackedAE.Get_Decoding_Result(number);
			float *pinput = myStackedAE.Get_Input(number);
			if (number == 0) {
				for (int k = 0; k < INPUT_DATA_DIMENSION; k++) {
					fprintf(myfile, "\t(%f, %f)", pinput[k], reprod[k]);
				}
			}
			else {
				for (int k = 0; k < eachEncoder_outDim[number]; k++) {
					fprintf(myfile, "\t(%f, %f)", pinput[k], reprod[k]);
				}
			}
			fprintf(myfile, "\n\n");
		}
	}
	fclose(myfile);

	return 0;
}
