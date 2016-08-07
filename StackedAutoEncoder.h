#ifndef __StackedAutoEncoder__
#define __StackedAutoEncoder__

class StackedAutoEncoder {
private:
	SimpleAutoEncoder* pAutoEncoder;
	int num_AE;

	void Delete();
public:
	StackedAutoEncoder() { pAutoEncoder = NULL; num_AE = 0; }
	StackedAutoEncoder(int number_AE, int inputDim, int* eachEncoder_outputDim);
	~StackedAutoEncoder() { Delete(); }

	void Init(int number_AE, int inputDim, int* eachEncoder_outputDim);
	int Is_Inited() { return pAutoEncoder != NULL; }

	

	void Back_Propagate(float* pInput, int action_idx); // train
	void Weight_Update(float learning_rate, int action_idx) { return pAutoEncoder[action_idx].Weight_Update(learning_rate); }


	SimpleAutoEncoder& operator[] (int idx) { return pAutoEncoder[idx]; }

	float* Get_Input(int idx) { return pAutoEncoder[idx].Get_Input(); }
	float* Get_Encoding_Result(int idx) { return pAutoEncoder[idx].Get_Encoding_Result(); }
	
	float* Get_Decoding_Result(int idx) { return pAutoEncoder[idx].Get_Decoding_Result(); }
	float Get_Reproduct_Error(int idx) { return pAutoEncoder[idx].Get_Decoding_Error(); }

	void Decoding(float* pInput, int action_idx);
	void Encoding(float* pInput, int action_idx);

};

#endif // !
