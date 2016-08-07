
#include "iostream"
#include "Layer.h"
#include "SimpleAutoEncoder.h"
#include "StackedAutoEncoder.h"

using namespace std;

StackedAutoEncoder::StackedAutoEncoder(int number_AE, int inputDim, int* eachEncoder_outputDim) {
	pAutoEncoder = NULL;
	num_AE = 0;

	Init(number_AE, inputDim, eachEncoder_outputDim);
}

void StackedAutoEncoder::Init(int number_AE, int inputDim, int* eachEncoder_outputDim) {
	if (Is_Inited())
		Delete();

	num_AE = number_AE;

	try {
		pAutoEncoder = new SimpleAutoEncoder[num_AE];
	}
	catch (bad_alloc& ba) {
		printf("bad memory allocation in func : %s, file : %s, line number : %d\n", __FUNCTION__, __FILE__, __LINE__);
		exit(-1);
	}

	pAutoEncoder[0].Init(inputDim, eachEncoder_outputDim[0]);
	for (int i = 1; i < num_AE; i++)
		pAutoEncoder[i].Init(eachEncoder_outputDim[i - 1], eachEncoder_outputDim[i]);

}

void StackedAutoEncoder::Delete() {
	if (pAutoEncoder != NULL) {
		delete[] pAutoEncoder;    //�̷��� �ϸ� pAutoEncoder[0] [1] [2] ������ pAutoEncoder[0].delete�� �Ҹ��ڸ� ���� ȣ��Ǵ��� Ȯ���ϱ�
		pAutoEncoder = NULL;
	}
	num_AE = 0;
}


//train �ϴ°�
//���߿� decode��� encode�� �Ǵ���... �ѹ� �׽�Ʈ�غ���//����Ұ� error����� decoder��������� ��
void StackedAutoEncoder::Back_Propagate(float* pInput, int action_idx) {
	if (action_idx == 0) {
		pAutoEncoder[0].Back_Propagate(pInput);
	}
	else {
		pAutoEncoder[0].Decoding(pInput);;
		for (int i = 1; i < action_idx; i++)
			pAutoEncoder[i].Decoding(pAutoEncoder[i - 1].Get_Encoding_Result());

		pAutoEncoder[action_idx].Back_Propagate(pAutoEncoder[action_idx - 1].Get_Encoding_Result());
	}
}


//���߿� decode��� encode�� �Ǵ���... �ѹ� �׽�Ʈ�غ���//����Ұ� error����� decoder��������� ��
void StackedAutoEncoder::Decoding(float* pInput, int action_idx) {
	if (action_idx == 0) {
		pAutoEncoder[0].Decoding(pInput);
	}
	else {
		pAutoEncoder[0].Decoding(pInput);
		for (int i = 1; i < action_idx; i++)
			pAutoEncoder[i].Decoding(pAutoEncoder[i - 1].Get_Encoding_Result());

		pAutoEncoder[action_idx].Decoding(pAutoEncoder[action_idx - 1].Get_Encoding_Result());
	}
}
//���߿� decode��� encode�� �Ǵ���... �ѹ� �׽�Ʈ�غ��� //����Ұ� error����� decoder��������� ��
void StackedAutoEncoder::Encoding(float* pInput, int action_idx) {
	if (action_idx == 0) {
		pAutoEncoder[0].Encoding(pInput);
	}
	else {
		pAutoEncoder[0].Decoding(pInput);
		for (int i = 1; i < action_idx; i++)
			pAutoEncoder[i].Decoding(pAutoEncoder[i - 1].Get_Encoding_Result());

		pAutoEncoder[action_idx].Encoding(pAutoEncoder[action_idx - 1].Get_Encoding_Result());
	}
}




