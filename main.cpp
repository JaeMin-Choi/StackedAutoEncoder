#include <iostream>
#include <fstream>
#include "Layer.h"
#include "SimpleAutoEncoder.h"

using namespace std;

#define MAX_EPOCH 1000000
#define MAX_ERROR 0.001f

#define INPUT_DATA_DIMENSION 3
#define TRAINING_DATA_SIZE 8

#define LEARNING_RATE 0.05f



int main(int argc, char* argv[]) {
	FILE* myfile = fopen("result.txt", "a");
	float train_data[8][INPUT_DATA_DIMENSION] = {
		{ 0,0,0},
		{0,0,1},
		{0,1,0},
		{0,1,1},
		{1,0,0},
		{1,0,1},
		{1,1,0},
		{1,1,1}
	};

	const int hidden_dim = 6;
	SimpleAutoEncoder myAE(INPUT_DATA_DIMENSION, hidden_dim);

	printf("\n\nStart Training Simple-Autoencoder \n");
	fprintf(myfile, "\n\nStart Training Simple-Autoencoder \n");

	for (int epoch = 0; epoch < MAX_EPOCH; epoch++) {
		float error = 0.f;

		for (int i = 0; i < TRAINING_DATA_SIZE; i++) {
			myAE.Back_Propagate(train_data[i]);
			error += myAE.Get_Decoding_Error();
		}

		error /= TRAINING_DATA_SIZE;

		myAE.Weight_Update(LEARNING_RATE);

		if ((epoch + 1) % 100 == 0) {
			printf("epoch = %d, error = %f \n", epoch + 1, error);
			fprintf(myfile, "epoch = %d, error = %f \n", epoch + 1, error);

			for (int j = 0; j < TRAINING_DATA_SIZE; j++) {
				myAE.Decoding(train_data[j]);
				float *decode_result = myAE.Get_Decoding_Result();



				printf("test%d: ( ", j);
				fprintf(myfile, "test%d: ( ", j);
				for (int k = 0; k < INPUT_DATA_DIMENSION; k++) {
					printf("%f ", train_data[j][k]);
					fprintf(myfile, "%f ", train_data[j][k]);
				}

				printf(") --> ( ");
				fprintf(myfile, ") --> ( ");
				for (int k = 0; k < INPUT_DATA_DIMENSION; k++) {
					printf("%f ", decode_result[k]);
					fprintf(myfile, "%f ", decode_result[k]);
				}
				printf(")\n");
				fprintf(myfile, ")\n");

			}
		}

		if (error < MAX_ERROR)
			break;
	}

	printf("Finish Training AutoEncoder  \n\n");
	fprintf(myfile, "Finish Training AutoEncoder  \n\n");

	printf("Test AutoEncoder\n");
	fprintf(myfile, "Test AutoEncoder\n");

	for (int j = 0; j < TRAINING_DATA_SIZE; j++) {
		myAE.Decoding(train_data[j]);
		float *encode_result = myAE.Get_Encoding_Result();
		float *decode_result = myAE.Get_Decoding_Result();

		printf("test%d: ( ", j);
		fprintf(myfile, "test%d: ( ", j);
		for (int k = 0; k < INPUT_DATA_DIMENSION; k++) {
			printf("%f ", train_data[j][k]);
			fprintf(myfile,"%f ", train_data[j][k]);
		}
		printf(") --> ( ");
		fprintf(myfile,") --> ( ");
		for (int k = 0; k < hidden_dim; k++) {
			printf("%f ", encode_result[k]);
			fprintf(myfile,"%f ", encode_result[k]);
		}
		printf(") --> ( ");
		fprintf(myfile,") --> ( ");
		for (int k = 0; k < INPUT_DATA_DIMENSION; k++) {
			printf("%f ", decode_result[k]);
			fprintf(myfile,"%f ", decode_result[k]);
		}
		printf(")\n");
		fprintf(myfile,")\n");
	}

	fclose(myfile);

	return 0;
}
