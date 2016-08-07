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
	aGradient = NULL; // pointer of gradient. Error를 weight으로 편미분 한 것, so the dimension will be same with pWeight's
	aDelta = NULL; // pointer of delta. Error를 net으로 미분한 것, so the dimension will be same with outputDim 
	aDeltaBar = NULL; // pointer of delta-bar. Error를 output으로 미분한 것, so the dimension will be same with outputDim
}

Layer::Layer(int inDim, int outDim) {
	inputDim = 0; // input dimension
	outputDim = 0; // output dimension

	pInput = NULL; // pointer of input
	aOutput = NULL; // pointer of output
	aWeight = NULL; // pointer of weight. The dimension will be "(inputDim+1)*outputDim". "+1" is added because of a bias.
	aGradient = NULL; // pointer of gradient. Error를 weight으로 편미분 한 것, so the dimension will be same with pWeight's
	aDelta = NULL; // pointer of delta. Error를 net으로 미분한 것, so the dimension will be same with outputDim 
	aDeltaBar = NULL; // pointer of delta-bar. Error를 output으로 미분한 것, so the dimension will be same with outputDim

	Init(inputDim, outputDim);
}

//layer init 해주는 함수  (default constructor 호출때만 이걸 call 하면 됨)
void Layer::Init(int input_dim, int output_dim) { //outut_dim 이 자기 레이어 노드 갯수가 된다. 
	if (Is_Inited())
		Delete();

	inputDim = input_dim;
	outputDim = output_dim;

	int weight_dim = (inputDim + 1)*outputDim; // weight 와 gradient는 bias 1개를 더 포함시켜야 함

	try {
		aOutput = new float[outputDim];
		aWeight = new float[weight_dim];
		aDelta = new float[outputDim];             // error를 net으로 미분한 것   --> output과 같은 차원이어야 함
		aDeltaBar = new float[outputDim];			// error을 output으로 미분한 것  --> output과 같은 차원이어야 함
		aGradient = new float[weight_dim];		// error을 weight으로 미분한 것   --> weight와 같은 차원이어야 함
	}
	catch (bad_alloc& ba) {
		printf("bad memory allocation in func : %s, file : %s, line number : %d\n", __FUNCTION__, __FILE__, __LINE__);
		exit(-1);
	}

	//gradient 배열 init
	//weight init
	//1-난수 생성시키는 방법
	random_device rd;
	mt19937_64 mt(rd());
	uniform_real_distribution<float> distribution(-1 / (float)sqrt(inputDim), 1 / (float)(sqrt(inputDim)));
	// init weights by random numbers in [ -1/root(input_dimension), 1/root(input_dimension) ] 
	for (int i = 0; i < weight_dim; i++) {
		aGradient[i] = 0.f;
		aWeight[i] = distribution(mt);
	}

	//output init
	//delta 배열 init
	//delta-bar 배열 init
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


//그러고 결과는 get_output
void Layer::Propagate(float* input_pointer) {
	pInput = input_pointer;
	for (int o = 0; o < outputDim; o++) {
		float net = 0.f;
		float* pWeight = aWeight + o * (inputDim + 1); // weight matrix의 low index 시작점
		for (int i = 0; i < inputDim; i++) {  // 얘가 돌면서 한 한 low를 다 계산함.
			net += pInput[i] * pWeight[i];
		}
		net += pWeight[inputDim]; // bias 더해주기

		aOutput[o] = Sigmoid(net);
	}

}
//weight matrix는 1차원이지만 2차원으로 생각해
// (input node,outputnode) 쌍임.  (1,1) (2,1) (3,1) ... (n,1) 
//	       				     	  (1,2) (2,2) (3,2) ....(n,2)
//									....
//					         	  (1,m) (2,m) (3,m) ... (n, m)
//			이게 그냥 일자로 쭉 나열되어 있다고 생각해


//deltabar는 Error을 output(NN의 최상단 layer의 output을 의미함)으로 미분한 것.
//error = MSE 
void Layer::Compute_Top_DeltaBar(float* pDesiredOutput) {	//최상단 layer의 deltabar 구하는 함수 (최상단 layer에서만 호출됨)
	for (int o = 0; o < outputDim; o++)
		aDeltaBar[o] = (aOutput[o] - pDesiredOutput[o]) / outputDim;
}



void Layer::Compute_Gradient() {	//gradient를 구하는 함수 
	int i = 0, o = 0;
	//delta는 Error을 net으로 미분한 것
	//Delta = Delta-Bar * sigmoid_differential(net) // 여기서 net은 output으로 대체될 수 있음
	for (o = 0; o < outputDim; o++)
		aDelta[o] = aDeltaBar[o] * Sigmoid_Differential(aOutput[o]);

	//위에서 delta를 다 구하고 이제 밑에서는 구한 delta를 가지고 gradient를 구한다.
	//gradient = delta * input으로 구할 수 있다.
	for (o = 0; o < outputDim; o++) {
		for (int i = 0; i < inputDim; i++) {
			aGradient[(inputDim + 1)*o + i] += aDelta[o] * pInput[i];
		}
		aGradient[(inputDim + 1) * o + inputDim] += aDelta[o]; // bias update
	}
}



void Layer::Compute_PrevDeltaBar(float* pPrevDeltaBar) {//전 layer의 deltabar를 구해주는 함수
														//이전 레이어의 delta-bar = 지금레이어의 delta * weight 로 구할 수 있다.
														// deltabar(n-1) = E를 output(n-1)로 나눈 것 = W(n)* delta(n)으로 정의될 수 있다. 
	for (int i = 0; i < inputDim; i++) {
		pPrevDeltaBar[i] = 0.f;
		for (int o = 0; o < outputDim; o++)
			pPrevDeltaBar[i] += aDelta[o] * aWeight[(inputDim + 1)*o + i];
	}
}


void Layer::Weight_Update(float learningRate) {//weight update는 w-learningRate*gradient 로 할수있음....
											   //float weight_length = 0.f; // added to make weight normalization

	for (int o = 0; o < outputDim; o++) {
		//	weight_length = 0.f; // added to make weight normalization
		for (int i = 0; i < inputDim + 1; i++) {
			aWeight[(inputDim + 1) * o + i] -= learningRate * aGradient[(inputDim + 1) * o + i];
			aGradient[(inputDim + 1) * o + i] = 0.f; //gradient 초기화
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
		aGradient[(inputDim + 1) * o + inputDim] = 0.f; //gradient 초기화
	}
}