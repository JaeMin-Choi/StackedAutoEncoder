#ifndef __Layer__
#define __Layer__

#ifndef TRUE
#define TRUE 1
#define FALSE 0
#endif // !TRUE


class Layer {
private:
	int inputDim; // input dimension
	int outputDim; // output dimension

	float* pInput; // pointer of input
	float* aOutput; // pointer of output

	float* aWeight; // pointer of weight. The dimension will be "(inputDim+1)*outputDim". "+1" is added because of a bias.
	float* aGradient; // pointer of gradient. Error�� weight���� ��̺� �� ��, so the dimension will be same with pWeight's
	float* aDelta; // pointer of delta. Error�� net���� �̺��� ��, so the dimension will be same with outputDim 
	float* aDeltaBar; // pointer of delta-bar. Error�� output���� �̺��� ��, so the dimension will be same with outputDim

	void Delete();
public:
	Layer();
	Layer(int inDim, int outDim);

	void Init(int inDim, int outDim); // Layer ������ ����, setting ���־�� �� �͵�.
	int Is_Inited() { return aWeight != NULL; } // Weight�� init������(!=null) 1, init�ȵ�����(==null) 0 return �Ѵ�.
	
	
	~Layer() { Delete(); }

	int Get_InputDim() { return inputDim; }
	int Get_OutputDim() { return outputDim; }
	float* Get_Input() { return pInput; }
	float* Get_Weight() { return aWeight; }
	float* Get_Gradient() { return aGradient; }
	float* Get_Output() { return aOutput; }
	float* Get_DeltaBar() { return aDeltaBar; } //delta-bar�� ������ �� �ִ� ������ ����

	void Propagate(float* input_pointer); //Propagate ���ִ� �Լ� 

	float Sigmoid(float net) { return 1.f / (1.f + (float)exp(-net)); } //activation function	
	float Sigmoid_Differential(float output) { return output * (1.f - output); } //sigmoid(net)�� ��ǲ���� ��

	void Compute_Top_DeltaBar(float* pDesiredOutput);//�ֻ�� layer�� deltabar ���ϴ� �Լ�
	void Compute_Gradient();	//gradient�� ���ϴ� �Լ� for NeuralNetwork
	void Compute_PrevDeltaBar(float* pPrevDeltaBar);//�� layer�� deltabar�� �����ִ� �Լ�
	void Weight_Update(float learningRate);//weight update�� w-learningRate*gradient �� �Ҽ�����....

	float Compute_Error(float* desired_output);

	void Bias_Update(float learningRate);
};
#endif // !__Layer__
