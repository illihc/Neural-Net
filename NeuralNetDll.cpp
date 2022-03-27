#include "pch.h"
#include "framework.h"
#include "NeuralNetDll.h"

//----------------------------- Initializing Class Neuron -----------------------------//

double Neuron::GetRandomWeight() 
{
	return rand() / double(RAND_MAX);
}

double Neuron::SigmoidFunc(double x)
{
	return 1 / (1 + exp(-x));
}

double Neuron::DerivativeSigmoidFunc(double x)
{
	return exp(-x) / (1 + 2 * exp(-x) + exp(2 * -x));
}

const double Neuron::SumDerivativesOfWeights(const Layer& NextLayer)
{
	double sum = 0.0;

	//Getting the sum of all influences, this neuron had on other neurons in the next Layer
	for (unsigned NeuronCount = 0; NeuronCount < NextLayer.size() - 1; NeuronCount++)
	{
		sum += OutputsWeights[NeuronCount].Weight * NextLayer[NeuronCount].gradient;
	}

	return sum;
}

Neuron::Neuron(unsigned NeuronOutputCount, unsigned _NeuronIndex)
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

void Neuron::CalculateNeuronValue(const Layer& PreviousLayer)
{
	double sum = 0.0;

	for (unsigned neuron = 0; neuron < PreviousLayer.size(); neuron++)
	{
		//Adding the product of the neurons, which feed this neuron and the weight of the edge between them
		sum += PreviousLayer[neuron].GetOutputValue() * PreviousLayer[neuron].OutputsWeights[NeuronIndex].Weight;
	}

	OutputValue = SigmoidFunc(sum);
}

void Neuron::SetOutputValue(double _OutputValue)
{
	OutputValue = _OutputValue;
	return;
}

void Neuron::CalculateOutputGradient(double TargetValue)
{
	//Because the TransferDerivativeFunction will be smallest on a local min. the network will work to get there
	//to keep the gradient as small as possible
	double delta = TargetValue - OutputValue;
	gradient = delta * DerivativeSigmoidFunc(OutputValue);
}

void Neuron::CalculateHiddenLayerGradient(const Layer& NextLayer)
{
	double DerivativesOfWeights = SumDerivativesOfWeights(NextLayer);
	gradient = DerivativesOfWeights * DerivativeSigmoidFunc(OutputValue);
}

void Neuron::UpdateInputWeight(Layer& PreviousLayer)
{
	for (unsigned NeuronCount = 0; NeuronCount < PreviousLayer.size(); NeuronCount++)
	{
		//Getting the deltaweight of the Edge connecting this neuron and the neuron in the layer left to it
		double OldWeightChange = PreviousLayer[NeuronCount].OutputsWeights[NeuronIndex].WeightChange;

		double NewWeightChange =
			//Add an input, which is modified by the training rate and the gradient of this neuron
			OverallNetLearningRate * PreviousLayer[NeuronCount].GetOutputValue() * gradient
			//add the direction, in which the neuron went last time, to secure it´s overall progress
			+ MomentumRate * OldWeightChange;

		//Setting the new deltaWeight
		PreviousLayer[NeuronCount].OutputsWeights[NeuronIndex].WeightChange = NewWeightChange;
		//Changing the old weiht, by the new deltaweight
		PreviousLayer[NeuronCount].OutputsWeights[NeuronIndex].Weight += NewWeightChange;
	}
}

double Neuron::GetOutputValue()const 
{ 
	return OutputValue; 
}

double Neuron::OverallNetLearningRate = 0.25;
double Neuron::MomentumRate = 0.75;


//----------------------------- Initializing Class NeuralNet -----------------------------//


void NeuralNet::CreateNet(vector<unsigned>& Layout)
{
	//Emptying the Layers, which might exist
	Layers.clear();

	//Getting the information, how many Layers there will be
	unsigned int LayerAmount = Layout.size();

	for (int LayerNumber = 0; LayerNumber < LayerAmount; LayerNumber++)
	{
		//Creating and adding a new Layer
		Layers.push_back(Layer());

		//Adding a variable, which holds the information in which Layer we are currently
		unsigned NeuronOutputCount;

		//If we are in the last layer, there are no outputs
		if (LayerNumber == (Layout.size() - 1))
			NeuronOutputCount = 0;
		//Otherwise the NeuronOUtputCount is the amount of Neurons in the next Layer
		else
			NeuronOutputCount = Layout[LayerNumber + 1];


		//Adding as many neurons, as the value of the current elemt of topology is plus a bias-neuron
		for (unsigned int NeuronNumber = 0; NeuronNumber <= Layout[LayerNumber]; NeuronNumber++)
		{
			//Getting the Layer, which was last added and adding a neuron to it
			Layers.back().push_back(Neuron(NeuronOutputCount, NeuronNumber));
		}

		//Set the bias neurons value to 1
		Layers.back().back().SetOutputValue(1);
	}
}

NeuralNet::NeuralNet(vector<unsigned>& Layout)
{
	CreateNet(Layout);
}

void NeuralNet::FeedForward(const vector<double>& InputValues)
{
	//Set all the Inputneurons to the Inputvalues 
	for (unsigned i = 0; i < InputValues.size(); i++)
	{
		Layers[0][i].SetOutputValue(InputValues[i]);
	}

	//Calculating all other neuronvalues starting with the second Layer (because the first was set above)
	for (unsigned LayerCount = 1; LayerCount < Layers.size(); LayerCount++)
	{
		//Getting the previous Layer
		Layer& PreviousLayer = Layers[LayerCount - 1];

		//looping through all neurons of the Layer, except the bias
		for (unsigned NeuronCount = 0; NeuronCount < Layers[LayerCount].size() - 1; NeuronCount++)
		{
			Layers[LayerCount][NeuronCount].CalculateNeuronValue(PreviousLayer);
		}
	}

	std::cout << "Have succesfully fed the net." << std::endl;
}

void NeuralNet::DoBackProp(const vector<double>& TargetValues)
{
	//Getting the outputlayer
	Layer& OutputLayer = Layers.back();

	//Calculating the error of all neurons in the outputlayer
	NetEstimateError = 0;
	for (unsigned neuroncount = 0; neuroncount < (OutputLayer.size() - 1); neuroncount++)
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


	//Calculate the gradient (Error-Rate) of the outputlayer
	for (unsigned NeuronCount = 0; NeuronCount < OutputLayer.size() - 1; NeuronCount++)
		OutputLayer[NeuronCount].CalculateOutputGradient(TargetValues[NeuronCount]);

	//Calculate the gradient (Error-Rate) of the hidden Layer(s), starting with the most right hidden layer
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
		for (unsigned NeuronCount = 0; NeuronCount < Layers[LayerCount].size() - 1; NeuronCount++)
		{
			Layers[LayerCount][NeuronCount].UpdateInputWeight(Layers[LayerCount - 1]);
		}
	}
}

const void NeuralNet::GetResult()
{
	//printing the output of each neuron in the last Layer to the console
	for (unsigned NeuronCount = 0; NeuronCount < Layers.back().size(); NeuronCount++)
	{
		//Not printing the bias neuron
		if (NeuronCount == Layers.back().size() - 1)
		{
			return;
		}

		std::cout << "OUtputneuronvalue is: " << Layers.back()[NeuronCount].GetOutputValue() << std::endl;
	}
}

void NeuralNet::SaveLayout(std::fstream& LayoutSaveFile)
{
	LayoutSaveFile.open("TopologySaveFile.txt", std::ios::out);
	LayoutSaveFile.clear();

	for (int LayerCount = 0; LayerCount < Layers.size(); LayerCount++)
	{
		if (LayoutSaveFile.is_open())
		{
			//Saving the Layout, ignoring the bias
			LayoutSaveFile << Layers[LayerCount].size() - 1 << "\n";
		}
	}

	LayoutSaveFile.close();
}

void NeuralNet::SaveNeuronValues(std::fstream& NeuronSaveFile, std::fstream& EdgesWeightSaveFile, std::fstream& EdgesWeightChangeSaveFile)
{
	NeuronSaveFile.open("NeuronSaveFile.txt", std::ios::out);
	EdgesWeightSaveFile.open("EdgesSaveFile.txt", std::ios::out);
	EdgesWeightChangeSaveFile.open("EdgesWeightChangeSaveFile.txt", std::ios::out);
	NeuronSaveFile.clear();
	EdgesWeightSaveFile.clear();
	EdgesWeightChangeSaveFile.clear();

	//For each Layer 
	for (int LayerCount = 0; LayerCount < Layers.size(); LayerCount++)
	{
		//And for each neuron in each layer
		for (int NeuronCount = 0; NeuronCount < Layers[LayerCount].size(); NeuronCount++)
		{
			//Write the neuronvalue to an SaveFile
			if (NeuronSaveFile.is_open())
				NeuronSaveFile << Layers[LayerCount][NeuronCount].GetOutputValue() << "\n";

			//For each weight the neuron has
			for (unsigned EdgeCount = 0; EdgeCount < Layers[LayerCount][NeuronCount].OutputsWeights.size(); EdgeCount++)
			{
				//Save the weight and weightchange of the edge to an file
				if (EdgesWeightSaveFile.is_open())
				{
					EdgesWeightSaveFile << Layers[LayerCount][NeuronCount].OutputsWeights[EdgeCount].Weight << "\n";
					EdgesWeightChangeSaveFile << Layers[LayerCount][NeuronCount].OutputsWeights[EdgeCount].WeightChange << "\n";
				}
			}

		}
	}

	NeuronSaveFile.close();
	EdgesWeightSaveFile.close();
	EdgesWeightChangeSaveFile.close();
}

void NeuralNet::LoadNeuronAndEdgeValues()
{
	std::fstream NeuronFile;
	std::fstream EdgeWeightFile;
	std::fstream EdgeWeightChangeFile;

	vector<double> NeuronValues;
	vector<double> EdgeWeightValues;
	vector<double> EdgeWeightChangeValues;

	//opening the files
	NeuronFile.open("NeuronSaveFile.txt", std::ios::in);
	EdgeWeightFile.open("EdgesSaveFile.txt", std::ios::in);
	EdgeWeightChangeFile.open("EdgesWeightChangeSaveFile.txt", std::ios::in);

	std::string FileLine;
	//reading the NeuronValues into the vector
	while (std::getline(NeuronFile, FileLine) && NeuronFile.is_open())
	{
		NeuronValues.push_back(std::stoi(FileLine));
	}
	//reading the edgeweightvalues into a the vector
	while (std::getline(EdgeWeightFile, FileLine) && EdgeWeightFile.is_open())
	{
		EdgeWeightValues.push_back(std::stoi(FileLine));
	}
	//reading the edgeweightchangevalues into a the vector
	while (std::getline(EdgeWeightChangeFile, FileLine) && EdgeWeightFile.is_open())
	{
		EdgeWeightChangeValues.push_back(std::stoi(FileLine));
	}

	unsigned NeuronValueCount = 0;
	unsigned EdgeValueCount = 0;

	//Looping through each layer, each neuron and each edge, to set their values
	for (unsigned LayerCount = 0; LayerCount < Layers.size(); LayerCount++)
	{
		for (unsigned NeuronCount = 0; NeuronCount < Layers[LayerCount].size(); NeuronCount++)
		{
			//Setting the current neurons value
			Layers[LayerCount][NeuronCount].SetOutputValue(NeuronValues[NeuronValueCount]);
			NeuronValueCount++;

			//looping through each edge, the neuron has
			for (unsigned EdgeCount = 0; EdgeCount < Layers[LayerCount][NeuronCount].OutputsWeights.size(); EdgeCount++)
			{
				Layers[LayerCount][NeuronCount].OutputsWeights[EdgeCount].Weight = EdgeWeightValues[EdgeValueCount];
				Layers[LayerCount][NeuronCount].OutputsWeights[EdgeCount].WeightChange = EdgeWeightChangeValues[EdgeValueCount];
				EdgeValueCount++;
			}
		}
	}
}


//----------------------------- Initializing all other functions -----------------------------//

vector<unsigned> Layout = { 11, 8, 3 };
NeuralNet Net(Layout);

void LoadNet()
{
	Net.LoadNeuronAndEdgeValues();
}

void SaveNet()
{
	std::fstream LayoutSaveFile;
	std::fstream NeuronSaveFile;
	std::fstream EdgesWeightSaveFile;
	std::fstream EdgesWeightChangeSaveFile;

	Net.SaveLayout(LayoutSaveFile);
	Net.SaveNeuronValues(NeuronSaveFile, EdgesWeightSaveFile, EdgesWeightChangeSaveFile);
}

void TrainNet(unsigned _TrainingSteps)
{
	std::cout << "Started Training." << std::endl;

	vector<double> InputValues;
	vector<double> OutputValues;
	vector<double> TargetValues;

	std::fstream InputValueFile;
	std::fstream TargetValueFile;

	TargetValueFile.open("TargetValueFile.txt", std::ios::in);
	InputValueFile.open("TrainingInputValues.txt", std::ios::in);

	if (!InputValueFile.is_open())
	{
		std::cout << "Error: InputValueFile isn´t opened!";
		return;
	}
	else if (!TargetValueFile.is_open())
	{
		std::cout << "Error: TargetValueFile isn´t opened!";
		return;
	}


	std::string FileLine;

	unsigned MaxTrainingsSteps = _TrainingSteps;
	unsigned CurrentTrainingStep = 0;
	unsigned const InputLineSkipper = 11;
	unsigned const OutputLineSkipper = 3;
	double LineValue;

	while (MaxTrainingsSteps > CurrentTrainingStep)
	{
		//The number the line we are currently reading 
		int LineNumber = 0;

		//-------------------Filling the InputvalueContainer---------------------
		LineNumber = 0;
		while (std::getline(InputValueFile, FileLine))
		{
			//std::cout << "InputValue is: " << FileLine << std::endl;
			LineNumber++;

			LineValue = std::stoi(FileLine);

			InputValues.push_back(LineValue);

			if (LineNumber == InputLineSkipper)
				break;
		}

		//---------------------Filling the TargetvalueContainer------------
		LineNumber = 0;
		while (std::getline(TargetValueFile, FileLine))
		{
			//Setting the linenumber to the correct value
			LineNumber++;


			//Adding the value as an Inputvalue
			TargetValues.push_back(std::stoi(FileLine));

			if (LineNumber == OutputLineSkipper)
				break;
		}


		Net.FeedForward(InputValues);
		Net.DoBackProp(TargetValues);
		Net.GetResult();

		//Clearing the ValueContainers
		InputValues.clear();
		TargetValues.clear();

		std::cout << "Ended Trainigsloop: " << CurrentTrainingStep << std::endl;
		CurrentTrainingStep++;
	}

	std::cout << "Ended Training." << std::endl;
}

void UseNet()
{
	vector<double> InputValues;
	std::fstream InputValueFile;
	std::string FileLine;

	InputValueFile.open("InputValueFile.txt", std::ios::in);

	if (!InputValueFile.is_open())
	{
		std::cout << "Error: InputValueFile isn´t opened!";
		return;
	}

	while (std::getline(InputValueFile, FileLine))
	{
		InputValues.push_back(std::stoi(FileLine));
	}

	Net.FeedForward(InputValues);
	Net.GetResult();
}
