#ifndef __SimpleAutoEncoder__
#define __SimpleAutoEncoder__


class SimpleAutoEncoder {
private:
	Layer* pEncoder;
	Layer* pDecoder;


	//encoder�� weight == decoder�� weight(t) 
	//weight�� tranpose���ִ� �Լ���.
	// ù��° ���̾��� weight�� tranpose�� ���� �ι�° ���̾��� weight�� �������ִ� �Լ�
	void Copy_Transpose_Weight();
	//simpleautoencoder�� back-pro �����, decoder�� gradient�� ���ϰ�, encoder�� gradient�� ���ؼ�, 
	//������ ���ؼ� weight update�� ���ִ°�
	void Get_Accumulated_Gradient();

	void Delete();

public:
	SimpleAutoEncoder() { pEncoder = NULL;	pDecoder = NULL; }
	SimpleAutoEncoder(int encode_inputDim, int encode_outputDim);
	~SimpleAutoEncoder() { Delete(); }

	void Init(int encode_inputDim, int encode_outputDim);
	int Is_Inited() { return (pEncoder != NULL && pDecoder != NULL); }

	
	float* Get_Encoding_Result() { return pEncoder->Get_Output(); }//encoder�� ��� getter
	float* Get_Decoding_Result() { return pDecoder->Get_Output(); } //decoder�� ��� getter
	float Get_Decoding_Error() { return pDecoder->Compute_Error(pEncoder->Get_Input()); } //autoencoder�� error 
	float* Get_Input() { return pEncoder->Get_Input(); }

	void Back_Propagate(float* pInput); // train
	void Weight_Update(float learning_rate);

	void Encoding(float* pInput) { pEncoder->Propagate(pInput); } // for stackedautoencoder test 
	void Decoding(float* pInput) { pEncoder->Propagate(pInput); pDecoder->Propagate(pEncoder->Get_Output()); }

};


#endif // !__SimpleAutoEncoder