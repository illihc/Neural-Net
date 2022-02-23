#include<vector>
#include<iostream>
#include<fstream>
#include<cstdlib>
#include<cassert>
#include<math.h>
#include<string>


using std::vector;

class Neuron;

//Defining a new kind of Variable "Layer", which contains a vector of neurons
typedef std::vector<Neuron> Layer;

struct Edge
{
	double Weight;
	double DeltaWeight;
};


//-------------------------------Neuron Class-------------------------------

class Neuron
{
private:
	double OutputValue;
	vector<Edge> OutputsWeights;
	unsigned int NeuronIndex;
	double gradient;
	static double OverallNetLearningRate; //Know as eta ("η"). Goes from 0 to 1
	static double MomentumRate; //Known as α. 

	static double GetRandomWeight()
	{
		//Get a random Value between 0 and 1
		return rand() / double(RAND_MAX);
	}

	static double TransferFunction(double x)
	{
		//Sigmoid function: Squeezes every input to a value between 0 and 1
		return 1 / (1 + exp(-x));
	}

	static double TransferDerivativeFunction(double x)
	{
		//derivative of the Sigmoid function
		return exp(-x) / (1 + 2 * exp(-x) + exp(2 * -x));
	}

	const double SumDOW(const Layer &NextLayer)
	{
		double sum = 0.0;

		//Getting the sum of all influences, this neuron had on other neurons in the next Layer
		for (unsigned NeuronCount = 0; NeuronCount < NextLayer.size() - 1; NeuronCount++)
		{
			sum += OutputsWeights[NeuronCount].Weight * NextLayer[NeuronCount].gradient;
		}

		return sum;
	}

public:
	Neuron(unsigned NeuronOutputCount, unsigned _NeuronIndex)
	{
		for (unsigned EdgeCount = 0; EdgeCount < NeuronOutputCount; EdgeCount++)
		{
			//Adding a new Edge to the Neuron
			OutputsWeights.push_back(Edge());

			//Asseign the Weight of the Edge to something random
			OutputsWeights.back().Weight = GetRandomWeight();
		}

		//Telling the neuron, which index it has in it´s layer
		NeuronIndex = _NeuronIndex;
	}

	void SetNeuronValue(const Layer& PreviousLayer) 
	{
		double sum = 0.0;

		for (unsigned neuron = 0; neuron < PreviousLayer.size(); neuron++)
		{
			//Adding the product of the neurons, which feed this neuron and the weight of the edge between them
			sum += PreviousLayer[neuron].GetOutputValue() * PreviousLayer[neuron].OutputsWeights[NeuronIndex].Weight;
		}

		OutputValue = TransferFunction(sum);
	}

	void SetOutputValue(double _OutputValue)
	{
		OutputValue = _OutputValue;
		return;
	}

	void CalculateOutputGradient(double TargetValue)
	{
		double delta = TargetValue - OutputValue;
		//Because the TransferDerivativeFunction will be smallest on a local min. the network will work to get there
		//to keep the gradient as small as possible
		gradient = delta * TransferDerivativeFunction(OutputValue);
	}

	void CalculateHiddenLayerGradient(const Layer &NextLayer)
	{
		double DerivativesOfWeights = SumDOW(NextLayer);
		gradient = DerivativesOfWeights * TransferDerivativeFunction(OutputValue);
	}

	void UpdateInputWeight(Layer &PreviousLayer)
	{
		for (unsigned NeuronCount = 0; NeuronCount < PreviousLayer.size(); NeuronCount++)
		{
			//Getting the deltaweight of the Edge connecting this neuron and the neuron in the layer left to it
			double OldDeltaWeight = PreviousLayer[NeuronCount].OutputsWeights[NeuronIndex].DeltaWeight;

			double NewDeltaWeight =
				//Add an input, which is modified by the training rate and the gradient of this neuron
				OverallNetLearningRate * PreviousLayer[NeuronCount].GetOutputValue() * gradient
				//add the direction, in which the neuron went last time, to secure it´s overall progress
				+ MomentumRate * OldDeltaWeight;

			//Setting the new deltaWeight
			PreviousLayer[NeuronCount].OutputsWeights[NeuronIndex].DeltaWeight = NewDeltaWeight;
			//Changing the old weiht, by the new deltaweight
			PreviousLayer[NeuronCount].OutputsWeights[NeuronIndex].Weight += NewDeltaWeight;
		}
	}

	 double GetOutputValue()const{return OutputValue;}
};

//Setting the value here, because for static member it won´t work inside a class
double Neuron::OverallNetLearningRate = 0.15;
double Neuron::MomentumRate = 0.5;


//--------------------------------------------Net Class------------------------------------


class NeuralNet
{
private:
	//Creating a Vector, which holds columns of Layers, of which each one contains a certain amount of neurons
	vector<Layer> Layers;
	double NetEstimateError = 0.0;
	double NetEstimateErrorAverage = 0.0;
	double ErrorSmothingFactor = 100.0;

public: 
	NeuralNet(vector<unsigned>& topology) 
	{
		//Getting the information, how many Layers there will be
		unsigned int LayerAmount = topology.size();

		for (int LayerNumber = 0; LayerNumber < LayerAmount; LayerNumber++)
		{
			//Creating and adding a new Layer
			Layers.push_back(Layer());
			std::cout << "Added a Layer" << std::endl;

			//Adding a variable, which holds the information in which Layer we are currently
			unsigned NeuronOutputCount;

			//If we are in the last layer, there are no outputs
			if(LayerNumber == (topology.size() -1))
				NeuronOutputCount = 0;
			//Otherwise the NeuronOUtputCount is the amount of Neurons in the next Layer
			else 
				NeuronOutputCount = topology[LayerNumber + 1];
			

			//Adding as many neurons, as the value of the current elemt of topology is plus a bias-neuron
			for (unsigned int NeuronNumber = 0; NeuronNumber <= topology[LayerNumber]; NeuronNumber++)
			{
				//Getting the Layer, which was last added and adding a neuron to it
				Layers.back().push_back(Neuron(NeuronOutputCount, NeuronNumber));
				std::cout << "Added a Neuron" << std::endl;
			}

			//Set the bias neurons value to 1
			Layers.back().back().SetOutputValue(1);
		}

	}

	//Putting inputs into the net
	void FeedForward(const vector<double>& InputValues) 
	{
		//Error checking
		assert(InputValues.size() == Layers[0].size() - 1);

		//Set all the Inputneurons to the Inputvalues 
		for (unsigned i = 0; i < InputValues.size(); i++)
		{
			Layers[0][i].SetOutputValue(InputValues[i]);
		}

		//Setting all other neuronvalues starting with the scound Layer (because the first was set above)
		for (unsigned LayerCount = 1; LayerCount < Layers.size(); LayerCount++)
		{
			//Getting the previous Layer
			Layer &PreviousLayer = Layers[LayerCount - 1];

			//looping through all neurons of the Layer, except the bias
			for (unsigned NeuronCount = 0; NeuronCount < Layers[LayerCount].size() -1; NeuronCount++)
			{
				Layers[LayerCount][NeuronCount].SetNeuronValue(PreviousLayer);
			}
		}
	}

	//Comparing the outputs of the individual neurons and edges, to the desired outputs
	void DoBackProp(const vector<double>& TargetValues) 
	{
		//Calculate all estimation errors of the net:

		//Getting the outputlayer
		Layer& OutputLayer = Layers.back();
		std::cout << "Output layer is: set" << std::endl;

		//Calculating the error of all neurons in the outputlayer
		NetEstimateError = 0;
		for (unsigned neuroncount = 0; neuroncount < OutputLayer.size() - 1; neuroncount++)
		{
			//Get the difference between the value the neuron has and the value it should have
			double delta = TargetValues[neuroncount] - OutputLayer[neuroncount].GetOutputValue();
			//Square that value 
			NetEstimateError += delta * delta;
		}

		//Get the average error in this layer
		NetEstimateError /= OutputLayer.size() - 1;
		//get the square root, because earllier the value of delta was squared
		NetEstimateError = sqrt(NetEstimateError);

		//Find the average error of multiple runs
		NetEstimateErrorAverage = (NetEstimateErrorAverage * ErrorSmothingFactor + NetEstimateError) / (ErrorSmothingFactor + 1.0);


		//Calculate the gradient of the outputlayer
		for (unsigned NeuronCount = 0; NeuronCount < OutputLayer.size() - 1; NeuronCount++)
			OutputLayer[NeuronCount].CalculateOutputGradient(TargetValues[NeuronCount]);

		//Calculate the gradient of the hidden Layer, starting with the most right hidden layer
		for (unsigned LayerCount = Layers.size() - 2; LayerCount > 0; LayerCount--)
		{
			//Getting each Neuron of each hidden Layer to calculate its Output
			for (unsigned NeuronCount = 0; NeuronCount < Layers[LayerCount].size(); NeuronCount++)
				Layers[LayerCount][NeuronCount].CalculateHiddenLayerGradient(Layers[LayerCount + 1]);
		}

		//Calculate all weights new Values
		for (unsigned LayerCount = Layers.size() - 1; LayerCount > 0; LayerCount--)
		{
			//Going through each enuron exept the bias
			for (unsigned NeuronCount = 0; NeuronCount < Layers[LayerCount].size() -1; NeuronCount++)
			{
				Layers[LayerCount][NeuronCount].UpdateInputWeight(Layers[LayerCount - 1]);
			}
		}

	}

	//Getting the output of the net
	const void GetResult(vector<double>& ResultValues)
	{
		//clearing the Resultvalues container
		ResultValues.clear();

		//Filling the container with the output of each neuron in the last Layer
		for (unsigned NeuronCount = 0; NeuronCount < Layers.back().size(); NeuronCount++)
		{
			ResultValues.push_back(Layers.back()[NeuronCount].GetOutputValue());
			std::cout << Layers.back()[NeuronCount].GetOutputValue() << std::endl;
		}
	}
};

int main()
{
	//The Lenght of topology holds the information how many columns of neurons = layers there will be
	//The value of each element of topology will hold the information, how many neurons there will be in the particulas layer
	vector<unsigned> Topology;
	vector<double> InputValues;
	vector<double> TargetValues;
	vector<double> ResultValues;

	std::fstream TopologyFile;
	std::fstream InputValueFile;
	std::fstream TargetValueFile;

	TopologyFile.open("TopologyFile.txt", std::ios::in);
	InputValueFile.open("InputValueFile.txt", std::ios::in);
	TargetValueFile.open("TargetValueFile.txt", std::ios::in);

	if (!TopologyFile.is_open())
	{
		std::cout << "Error: TopologyFile isn´t opened!";
		return 1;
	}
	else if (!InputValueFile.is_open())
	{
		std::cout << "Error: InputValueFile isn´t opened!";
		return 1;
	}
	else if (!TargetValueFile.is_open())
	{
		std::cout << "Error: TargetValueFile isn´t opened!";
		return 1;
	}


	std::string FileLine;

	//Filling topology with the needed information from the file
	while (std::getline(TopologyFile, FileLine))
	{
		Topology.push_back(std::stoi(FileLine));
	}

	//Saving the input values
	while (std::getline(InputValueFile, FileLine))
	{
		InputValues.push_back(std::stoi(FileLine));
	}

	//Saving the Target values
	while (std::getline(TargetValueFile, FileLine))
	{
		TargetValues.push_back(std::stoi(FileLine));
	}

	//Creating the Neural Net
	NeuralNet Net(Topology);

	//give the information to the net
	Net.FeedForward(InputValues);
	std::cout << "Feed forward done!" << std::endl;
	Net.DoBackProp(TargetValues);
	std::cout << "Backprop done!" << std::endl;
	Net.GetResult(ResultValues);
	
	/*for (int i = 0; i < ResultValues.size(); i++)
	{
		std::cout << ResultValues[i] << std::endl;
	}
	*/
	return 0;
}
