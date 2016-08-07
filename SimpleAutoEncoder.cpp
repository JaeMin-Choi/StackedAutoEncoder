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



//gradient 계산하는 것
void SimpleAutoEncoder::Back_Propagate(float* pInput) {
	Decoding(pInput);

	pDecoder->Compute_Top_DeltaBar(pInput);
	pDecoder->Compute_Gradient();

	pDecoder->Compute_PrevDeltaBar(pEncoder->Get_DeltaBar());
	pEncoder->Compute_Gradient();

}


//구한 gradient로 첫번째 레이어(layer[0])의 weight을 update하고
//그 weight의 tranpose를 취하여 layer[1]에 assign 해준다
void SimpleAutoEncoder::Weight_Update(float learning_rate) {
	Get_Accumulated_Gradient(); //decoder의 gradient여기서 초기화 됨(bias gradient 제외)
	pEncoder->Weight_Update(learning_rate); // encoder의 gradient여기서 초기화 됨
	Copy_Transpose_Weight();
	pDecoder->Bias_Update(learning_rate); // decoder의 bias gradient여기서 초기화 됨

}


// encoder weight의 tranpose를 decoder weight에 복사해주는 함수
//weight update 하는 함수 호출 하면 encoder의 weight가 update될 거야,
//encoder의 weight update를 한 뒤, 그 weight를 tranpose해서 decoder에 해줌.
void SimpleAutoEncoder::Copy_Transpose_Weight() {
	int encoder_outputDim = pEncoder->Get_OutputDim();
	int encoder_inputDim = pEncoder->Get_InputDim();
	float* pEncoder_weight = pEncoder->Get_Weight();
	float* pDecoder_weight = pDecoder->Get_Weight();

	//이렇게 하면 encoder의 weight transpose가 decoder의 weight로 저장이 됨.
	for (int i = 0; i < encoder_outputDim; i++) {
		for (int j = 0; j < encoder_inputDim; j++) {
			pDecoder_weight[(encoder_outputDim + 1)*j + i] = pEncoder_weight[(encoder_inputDim + 1)*i + j];
		}
	}
}


//encoder, decoder의 두 gradient를 더해줘 (decoder의 gradient를 transpose해서)
//encoder gradient에 저장함
//그런다음 decoder의 gradient를 0으로 초기화 해줌
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
