#ifndef __SimpleAutoEncoder__
#define __SimpleAutoEncoder__


class SimpleAutoEncoder {
private:
	Layer* pEncoder;
	Layer* pDecoder;


	//encoder의 weight == decoder의 weight(t) 
	//weight를 tranpose해주는 함수임.
	// 첫번째 레이어의 weight의 tranpose를 취해 두번째 레이어의 weight에 복사해주는 함수
	void Copy_Transpose_Weight();
	//simpleautoencoder의 back-pro 방식은, decoder의 gradient를 구하고, encoder의 gradient를 구해서, 
	//각각을 더해서 weight update를 해주는것
	void Get_Accumulated_Gradient();

	void Delete();

public:
	SimpleAutoEncoder() { pEncoder = NULL;	pDecoder = NULL; }
	SimpleAutoEncoder(int encode_inputDim, int encode_outputDim);
	~SimpleAutoEncoder() { Delete(); }

	void Init(int encode_inputDim, int encode_outputDim);
	int Is_Inited() { return (pEncoder != NULL && pDecoder != NULL); }

	
	float* Get_Encoding_Result() { return pEncoder->Get_Output(); }//encoder의 결과 getter
	float* Get_Decoding_Result() { return pDecoder->Get_Output(); } //decoder의 결과 getter
	float Get_Decoding_Error() { return pDecoder->Compute_Error(pEncoder->Get_Input()); } //autoencoder의 error 
	float* Get_Input() { return pEncoder->Get_Input(); }

	void Back_Propagate(float* pInput); // train
	void Weight_Update(float learning_rate);

	void Encoding(float* pInput) { pEncoder->Propagate(pInput); } // for stackedautoencoder test 
	void Decoding(float* pInput) { pEncoder->Propagate(pInput); pDecoder->Propagate(pEncoder->Get_Output()); }

};


#endif // !__SimpleAutoEncoder