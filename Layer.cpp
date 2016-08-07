#include <math.h>
#include <random>

#include "Layer.h"

using namespace std;

Layer::Layer() {
	inputDim = 0; // input dimension
	outputDim = 0; // output dimension

	pInput = NULL; // pointer of input
	aOutput = NULL; // pointer of output
	aWeight = NULL; // pointer of weight. The dimension will be "(inputDim+1)*outputDim". "+1" is added because of a bias.
	aGradient = NULL; // pointer of gradient. Error�� weight���� ��̺� �� ��, so the dimension will be same with pWeight's
	aDelta = NULL; // pointer of delta. Error�� net���� �̺��� ��, so the dimension will be same with outputDim 
	aDeltaBar = NULL; // pointer of delta-bar. Error�� output���� �̺��� ��, so the dimension will be same with outputDim
}

Layer::Layer(int inDim, int outDim) {
	inputDim = 0; // input dimension
	outputDim = 0; // output dimension

	pInput = NULL; // pointer of input
	aOutput = NULL; // pointer of output
	aWeight = NULL; // pointer of weight. The dimension will be "(inputDim+1)*outputDim". "+1" is added because of a bias.
	aGradient = NULL; // pointer of gradient. Error�� weight���� ��̺� �� ��, so the dimension will be same with pWeight's
	aDelta = NULL; // pointer of delta. Error�� net���� �̺��� ��, so the dimension will be same with outputDim 
	aDeltaBar = NULL; // pointer of delta-bar. Error�� output���� �̺��� ��, so the dimension will be same with outputDim

	Init(inputDim, outputDim);
}

//layer init ���ִ� �Լ�  (default constructor ȣ�⶧�� �̰� call �ϸ� ��)
void Layer::Init(int input_dim, int output_dim) { //outut_dim �� �ڱ� ���̾� ��� ������ �ȴ�. 
	if (Is_Inited())
		Delete();

	inputDim = input_dim;
	outputDim = output_dim;

	int weight_dim = (inputDim + 1)*outputDim; // weight �� gradient�� bias 1���� �� ���Խ��Ѿ� ��

	try {
		aOutput = new float[outputDim];
		aWeight = new float[weight_dim];
		aDelta = new float[outputDim];             // error�� net���� �̺��� ��   --> output�� ���� �����̾�� ��
		aDeltaBar = new float[outputDim];			// error�� output���� �̺��� ��  --> output�� ���� �����̾�� ��
		aGradient = new float[weight_dim];		// error�� weight���� �̺��� ��   --> weight�� ���� �����̾�� ��
	}
	catch (bad_alloc& ba) {
		printf("bad memory allocation in func : %s, file : %s, line number : %d\n", __FUNCTION__, __FILE__, __LINE__);
		exit(-1);
	}

	//gradient �迭 init
	//weight init
	//1-���� ������Ű�� ���
	random_device rd;
	mt19937_64 mt(rd());
	uniform_real_distribution<float> distribution(-1 / (float)sqrt(inputDim), 1 / (float)(sqrt(inputDim)));
	// init weights by random numbers in [ -1/root(input_dimension), 1/root(input_dimension) ] 
	for (int i = 0; i < weight_dim; i++) {
		aGradient[i] = 0.f;
		aWeight[i] = distribution(mt);
	}

	//output init
	//delta �迭 init
	//delta-bar �迭 init
	for (int i = 0; i < outputDim; i++) {
		aOutput[i] = 0.f;
		aDelta[i] = 0.f;
		aDeltaBar[i] = 0.f;
	}

}
void Layer::Delete() {
	if (aOutput != NULL) {
		delete[] aOutput;
		aOutput = NULL;
	}
	if (aWeight != NULL) {
		delete[] aWeight;
		aWeight = NULL;
	}
	if (aGradient != NULL) {
		delete[] aGradient;
		aGradient = NULL;
	}
	if (aDelta != NULL) {
		delete[] aDelta;
		aDelta = NULL;
	}
	if (aDeltaBar != NULL) {
		delete[] aDeltaBar;
		aDeltaBar = NULL;
	}

	if (pInput != NULL) {
		pInput = NULL;
	}
	inputDim = 0;
	outputDim = 0;

}


//�׷��� ����� get_output
void Layer::Propagate(float* input_pointer) {
	pInput = input_pointer;
	for (int o = 0; o < outputDim; o++) {
		float net = 0.f;
		float* pWeight = aWeight + o * (inputDim + 1); // weight matrix�� low index ������
		for (int i = 0; i < inputDim; i++) {  // �갡 ���鼭 �� �� low�� �� �����.
			net += pInput[i] * pWeight[i];
		}
		net += pWeight[inputDim]; // bias �����ֱ�

		aOutput[o] = Sigmoid(net);
	}

}
//weight matrix�� 1���������� 2�������� ������
// (input node,outputnode) ����.  (1,1) (2,1) (3,1) ... (n,1) 
//	       				     	  (1,2) (2,2) (3,2) ....(n,2)
//									....
//					         	  (1,m) (2,m) (3,m) ... (n, m)
//			�̰� �׳� ���ڷ� �� �����Ǿ� �ִٰ� ������


//deltabar�� Error�� output(NN�� �ֻ�� layer�� output�� �ǹ���)���� �̺��� ��.
//error = MSE 
void Layer::Compute_Top_DeltaBar(float* pDesiredOutput) {	//�ֻ�� layer�� deltabar ���ϴ� �Լ� (�ֻ�� layer������ ȣ���)
	for (int o = 0; o < outputDim; o++)
		aDeltaBar[o] = (aOutput[o] - pDesiredOutput[o]) / outputDim;
}



void Layer::Compute_Gradient() {	//gradient�� ���ϴ� �Լ� 
	int i = 0, o = 0;
	//delta�� Error�� net���� �̺��� ��
	//Delta = Delta-Bar * sigmoid_differential(net) // ���⼭ net�� output���� ��ü�� �� ����
	for (o = 0; o < outputDim; o++)
		aDelta[o] = aDeltaBar[o] * Sigmoid_Differential(aOutput[o]);

	//������ delta�� �� ���ϰ� ���� �ؿ����� ���� delta�� ������ gradient�� ���Ѵ�.
	//gradient = delta * input���� ���� �� �ִ�.
	for (o = 0; o < outputDim; o++) {
		for (int i = 0; i < inputDim; i++) {
			aGradient[(inputDim + 1)*o + i] += aDelta[o] * pInput[i];
		}
		aGradient[(inputDim + 1) * o + inputDim] += aDelta[o]; // bias update
	}
}



void Layer::Compute_PrevDeltaBar(float* pPrevDeltaBar) {//�� layer�� deltabar�� �����ִ� �Լ�
														//���� ���̾��� delta-bar = ���ݷ��̾��� delta * weight �� ���� �� �ִ�.
														// deltabar(n-1) = E�� output(n-1)�� ���� �� = W(n)* delta(n)���� ���ǵ� �� �ִ�. 
	for (int i = 0; i < inputDim; i++) {
		pPrevDeltaBar[i] = 0.f;
		for (int o = 0; o < outputDim; o++)
			pPrevDeltaBar[i] += aDelta[o] * aWeight[(inputDim + 1)*o + i];
	}
}


void Layer::Weight_Update(float learningRate) {//weight update�� w-learningRate*gradient �� �Ҽ�����....
											   //float weight_length = 0.f; // added to make weight normalization

	for (int o = 0; o < outputDim; o++) {
		//	weight_length = 0.f; // added to make weight normalization
		for (int i = 0; i < inputDim + 1; i++) {
			aWeight[(inputDim + 1) * o + i] -= learningRate * aGradient[(inputDim + 1) * o + i];
			aGradient[(inputDim + 1) * o + i] = 0.f; //gradient �ʱ�ȭ
													 //		weight_length += (float)pow(aWeight[(inputDim + 1) * i + j], 2); // added to make weight normalization
		}

		//	weight_length = (float)sqrt(weight_length);	// added to make weight normalization

		//	for (int j = 0; j < inputDim+1; j++) {	// added to make weight normalization
		//		aWeight[(inputDim + 1) * i + j] /= weight_length;	// added to make weight normalization
		//	}	
	}
}

//EMS == (output - desired_output)^2 / (2*outputDim)
float Layer::Compute_Error(float* desired_output) {
	float error = 0.f;
	for (int o = 0; o < outputDim; o++)
		error += (aOutput[o] - desired_output[o]) * (aOutput[o] - desired_output[o]);

	error /= (2 * outputDim);

	return error;
}


void Layer::Bias_Update(float learningRate) {
	for (int o = 0; o < outputDim; o++) {
		aWeight[(inputDim + 1) * o + inputDim] -= learningRate * aGradient[(inputDim + 1) * o + inputDim];
		aGradient[(inputDim + 1) * o + inputDim] = 0.f; //gradient �ʱ�ȭ
	}
}