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
	float* aGradient; // pointer of gradient. Error를 weight으로 편미분 한 것, so the dimension will be same with pWeight's
	float* aDelta; // pointer of delta. Error를 net으로 미분한 것, so the dimension will be same with outputDim 
	float* aDeltaBar; // pointer of delta-bar. Error를 output으로 미분한 것, so the dimension will be same with outputDim

	void Delete();
public:
	Layer();
	Layer(int inDim, int outDim);

	void Init(int inDim, int outDim); // Layer 생성될 때에, setting 해주어야 할 것들.
	int Is_Inited() { return aWeight != NULL; } // Weight가 init됐으면(!=null) 1, init안됐으면(==null) 0 return 한다.
	
	
	~Layer() { Delete(); }

	int Get_InputDim() { return inputDim; }
	int Get_OutputDim() { return outputDim; }
	float* Get_Input() { return pInput; }
	float* Get_Weight() { return aWeight; }
	float* Get_Gradient() { return aGradient; }
	float* Get_Output() { return aOutput; }
	float* Get_DeltaBar() { return aDeltaBar; } //delta-bar를 참조할 수 있는 포인터 제공

	void Propagate(float* input_pointer); //Propagate 해주는 함수 

	float Sigmoid(float net) { return 1.f / (1.f + (float)exp(-net)); } //activation function	
	float Sigmoid_Differential(float output) { return output * (1.f - output); } //sigmoid(net)이 인풋으로 들어가

	void Compute_Top_DeltaBar(float* pDesiredOutput);//최상단 layer의 deltabar 구하는 함수
	void Compute_Gradient();	//gradient를 구하는 함수 for NeuralNetwork
	void Compute_PrevDeltaBar(float* pPrevDeltaBar);//전 layer의 deltabar를 구해주는 함수
	void Weight_Update(float learningRate);//weight update는 w-learningRate*gradient 로 할수있음....

	float Compute_Error(float* desired_output);

	void Bias_Update(float learningRate);
};
#endif // !__Layer__
