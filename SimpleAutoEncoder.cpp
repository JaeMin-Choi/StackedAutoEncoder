#include <iostream>

#include "Layer.h"
#include "SimpleAutoEncoder.h"

using namespace std;

SimpleAutoEncoder::SimpleAutoEncoder(int encode_inputDim, int encode_outputDim) {
	pEncoder = NULL;
	pDecoder = NULL;

	Init(encode_inputDim, encode_outputDim);
}

void SimpleAutoEncoder::Init(int encode_inputDim, int encode_outputDim) {
	if (Is_Inited())
		Delete();

	try {
		pEncoder = new Layer;
		pDecoder = new Layer;
	}
	catch (bad_alloc& ba) {
		printf("bad memory allocation in func : %s, file : %s, line number : %d\n", __FUNCTION__, __FILE__, __LINE__);
		exit(-1);
	}

	pEncoder->Init(encode_inputDim, encode_outputDim);
	pDecoder->Init(encode_outputDim, encode_inputDim);

	Copy_Transpose_Weight();
}

void SimpleAutoEncoder::Delete() {
	if (pEncoder != NULL) {
		delete pEncoder;
		pEncoder = NULL;
	}

	if (pDecoder != NULL) {
		delete pDecoder;
		pDecoder = NULL;
	}
}



//gradient ����ϴ� ��
void SimpleAutoEncoder::Back_Propagate(float* pInput) {
	Decoding(pInput);

	pDecoder->Compute_Top_DeltaBar(pInput);
	pDecoder->Compute_Gradient();

	pDecoder->Compute_PrevDeltaBar(pEncoder->Get_DeltaBar());
	pEncoder->Compute_Gradient();

}


//���� gradient�� ù��° ���̾�(layer[0])�� weight�� update�ϰ�
//�� weight�� tranpose�� ���Ͽ� layer[1]�� assign ���ش�
void SimpleAutoEncoder::Weight_Update(float learning_rate) {
	Get_Accumulated_Gradient(); //decoder�� gradient���⼭ �ʱ�ȭ ��(bias gradient ����)
	pEncoder->Weight_Update(learning_rate); // encoder�� gradient���⼭ �ʱ�ȭ ��
	Copy_Transpose_Weight();
	pDecoder->Bias_Update(learning_rate); // decoder�� bias gradient���⼭ �ʱ�ȭ ��

}


// encoder weight�� tranpose�� decoder weight�� �������ִ� �Լ�
//weight update �ϴ� �Լ� ȣ�� �ϸ� encoder�� weight�� update�� �ž�,
//encoder�� weight update�� �� ��, �� weight�� tranpose�ؼ� decoder�� ����.
void SimpleAutoEncoder::Copy_Transpose_Weight() {
	int encoder_outputDim = pEncoder->Get_OutputDim();
	int encoder_inputDim = pEncoder->Get_InputDim();
	float* pEncoder_weight = pEncoder->Get_Weight();
	float* pDecoder_weight = pDecoder->Get_Weight();

	//�̷��� �ϸ� encoder�� weight transpose�� decoder�� weight�� ������ ��.
	for (int i = 0; i < encoder_outputDim; i++) {
		for (int j = 0; j < encoder_inputDim; j++) {
			pDecoder_weight[(encoder_outputDim + 1)*j + i] = pEncoder_weight[(encoder_inputDim + 1)*i + j];
		}
	}
}


//encoder, decoder�� �� gradient�� ������ (decoder�� gradient�� transpose�ؼ�)
//encoder gradient�� ������
//�׷����� decoder�� gradient�� 0���� �ʱ�ȭ ����
void SimpleAutoEncoder::Get_Accumulated_Gradient() {
	int encoder_outputDim = pEncoder->Get_OutputDim();
	int encoder_inputDim = pEncoder->Get_InputDim();
	float* pEncoder_gradient = pEncoder->Get_Gradient();
	float* pDecoder_gradient = pDecoder->Get_Gradient();
	for (int i = 0; i < encoder_outputDim; i++) {
		for (int j = 0; j < encoder_inputDim; j++) {
			pEncoder_gradient[(encoder_inputDim + 1) * i + j] += pDecoder_gradient[(encoder_outputDim + 1)*j + i];
			pDecoder_gradient[(encoder_outputDim + 1) * j + i] = 0.f;
		}
	}
}
