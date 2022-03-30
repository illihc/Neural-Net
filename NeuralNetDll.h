#pragma once
#include<vector>
#include<iostream>
#include<fstream>
#include<cstdlib>
#include<math.h>
#include<string>

#ifdef NEURALNETDLL_EXPORTS
#define NEURALNETDLL_API __declspec(dllexport)
#else
#define NEURALNETDLL_API __declspec(dllimport)
#endif


using std::vector;

extern "C" class NEURALNETDLL_API Neuron;

//Defining a new kind of Variable "Layer", which contains a vector of neurons
extern "C" NEURALNETDLL_API typedef std::vector<Neuron> Layer;

extern "C" struct NEURALNETDLL_API Edge
{
	double Weight;
	double WeightChange;
};


//-------------------------------Neuron Class-------------------------------//

extern "C" class NEURALNETDLL_API Neuron
{
private:
	double OutputValue;
	unsigned int NeuronIndex;
	double gradient;
	static double OverallNetLearningRate;
	static double MomentumRate;

	//Get a random Value between 0 and 1
	static double GetRandomWeight();

	//Sigmoid function: Squeezes every input to a value between 0 and 1
	static double SigmoidFunc(double x);

	//The derivative of the Sigmoid function
	static double DerivativeSigmoidFunc(double x);

	//Sums up all the derivatives of the neurons in the last layer
	const double SumDerivativesOfWeights(const Layer& NextLayer);

public:
	//A list of Edges the neuron has to its right
	vector<Edge> OutputsWeights;

	//Create the neuron with edges to all neurons right to it and an index to determine its position in its layer
	Neuron(unsigned NeuronOutputCount, unsigned _NeuronIndex);

	//Calculate the Output of this neuron, if it is not in the Inpulayer and not a bias
	void CalculateNeuronValue(const Layer& PreviousLayer);

	//Set the Outputvalue if this neuron is in the Inputlayer or a bias
	void SetOutputValue(double _OutputValue);

	//Calculating the Error-Rate of the Outputneuron
	void CalculateOutputGradient(double TargetValue);

	//Calculate the Error-Rate of the neuron in a hidden Layer
	void CalculateHiddenLayerGradient(const Layer& NextLayer);

	//Update the weight of the edges, connecting this neuron to the ones in the layer before (left to it)
	void UpdateInputWeight(Layer& PreviousLayer);

	//Get the Outputvalue of the Neuron
	double GetOutputValue() const;
};


//--------------------------------------------Net Class------------------------------------//


extern "C" class NEURALNETDLL_API NeuralNet
{
private:
	//Creating a Vector, which holds columns of Layers, of which each one contains a certain amount of neurons
	vector<Layer> Layers;

	double NetEstimateError = 0.0;
	double NetEstimateErrorAverage = 0.0;
	double ErrorSmothingFactor = 100.0;

	//Create a new Net
	void CreateNet(vector<unsigned>& Layout);

public:

	NeuralNet(vector<unsigned>& Layout);

	//Putting inputs into the net
	void FeedForward(const vector<double>& InputValues);

	//Comparing the outputs of the individual neurons and edges, to the desired outputs 
	//Change the Neuron and Edgevalues then, to get a better result
	void DoBackProp(const vector<double>& TargetValues);

	//Getting the output of the net
	const double GetResult(unsigned NeuronCount);

	//save the current layout of the net
	void SaveLayout(std::fstream& LayoutSaveFile);

	//Save the current Neuron- and edgevalues
	void SaveNeuronValues(std::fstream& NeuronSaveFile, std::fstream& EdgesWeightSaveFile, std::fstream& EdgesWeightChangeSaveFile);

	//Load the Values of everything of an File
	void LoadNeuronAndEdgeValues();
};

//-------------------------------------------other methods--------------------------------------//

extern "C" NEURALNETDLL_API void LoadNet();

extern "C" NEURALNETDLL_API void TrainNet(unsigned _TrainingSteps);

extern "C" NEURALNETDLL_API void UseNet();

extern "C" NEURALNETDLL_API void SaveNet();

extern "C" NEURALNETDLL_API double GetNetResult(unsigned ParameterCount);

