#include "Experiment.h"
#include "dataset/ImageBin.h"
#include "stdp/Multiplicative.h"
#include "stdp/Biological.h"
#include "layer/Convolution.h"
#include "layer/Pooling.h"
#include "Distribution.h"
#include "execution/TrainingExecution.h"
#include "execution/TrainingSparseExecution.h"
#include "analysis/SaveFeatureNumpy.h"
#include "analysis/Activity.h"
#include "analysis/Coherence.h"
#include "process/Input.h"
#include "process/Scaling.h"
#include "process/Pooling.h"
#include "process/GrayScale.h"
#include "process/OnOffFilter.h"
#include "process/WhitenPatchesLoader.h"
#include "process/SeparateSign.h"
#include "dep/ArduinoJson-v6.17.3.h"

// NOTE: Works on Linux only
std::string get_build_path()
{
	// Get the absolute path to the executable
	char self[4096] = {0};
	int nchar = readlink("/proc/self/exe", self, sizeof self);
	std::string path = std::string(self);
	// Remove the name of the executable from the path
	std::size_t pos = path.find_last_of("/");
	if (pos != std::string::npos)
	{
		path = path.substr(0, pos);
	}
	return path;
}


void load_dataset(AbstractExperiment *experiment, std::string &data_path, std::string &label_path, std::string &dataset)
{
	int width = 0;
	int height = 0;
	int depth = 0;
	if (dataset == "MNIST")
	{
		width = 28;
		height = 28;
		depth = 1;
	}
	else if (dataset == "CIFAR10")
	{
		width = 32;
		height = 32;
		depth = 3;
	}
	else if (dataset == "STL10")
	{
		width = 96;
		height = 96;
		depth = 3;
	}
	else if (dataset == "STL10-64")
	{
		width = 64;
		height = 64;
		depth = 3;
	}
	else if (dataset == "ETH80")
	{
		width = 100;
		height = 100;
		depth = 3;
	}
	else
	{
		throw std::runtime_error("Dataset loader for " + dataset + " is not implemented");
	}
	experiment->add_train<dataset::ImageBin>(data_path, label_path, width, height, depth, dataset);
}

int main(int argc, char **argv)
{

	// Argument parsing
	if (argc < 4 || argc > 6)
	{
		throw std::runtime_error("Usage: " + std::string(argv[0]) + " <DATA_PATH> <LABEL_PATH> <CONFIG_PATH> [ <OUTPUT_PATH = ./> ] [ <SEED = 0> ]");
	}
	std::string data_path = std::string(argv[1]);
	std::string label_path = std::string(argv[2]);
	std::string config_path = std::string(argv[3]);
	std::string output_path = "./";
	if (argc > 4)
	{
		output_path = std::string(argv[4]);
	}
	int seed = 0;
	if (argc > 5)
	{
		seed = std::stoi(argv[5]);
	}

	// Load config
	std::ifstream _jsonTextFile(config_path);
	if (!_jsonTextFile.good())
	{
		throw std::runtime_error("Failed to open JSON config");
	}
	std::stringstream buffer;
	buffer << _jsonTextFile.rdbuf();
	std::string _jsonText = buffer.str();
	_jsonTextFile.close();
	DynamicJsonDocument config(JSON_ARRAY_SIZE(_jsonText.length()));
	DeserializationError error = deserializeJson(config, _jsonText.c_str());
	if (error)
	{
		throw std::runtime_error("Failed to parse JSON config");
	}
	// Add seed to config
	config["seed"] = seed;

	// Initialize experiment
	AbstractExperiment *experiment;
	std::string exp_name = "train";
	std::string model_path = output_path + "/model/";
	if (config["use_sparse"] == true)
	{
		experiment = new Experiment<TrainingSparseExecution>(argc, argv, output_path, model_path, exp_name, seed, true);
	}
	else
	{
		experiment = new Experiment<TrainingExecution>(argc, argv, output_path, model_path, exp_name, seed, true);
	}

	// Save config path to the model directory
	std::ofstream output_config_file(model_path + "/config.json");
	std::string jsonString;
	serializeJsonPretty(config, jsonString);
	output_config_file << jsonString;
	output_config_file.close();

	// Load dataset
	std::string dataset_name(config["dataset"].as<const char *>());
	load_dataset(experiment, data_path, label_path, dataset_name);

	// Preprocessing
	if (config.containsKey("whiten") && config["whiten"] == true)
	{
		experiment->push<process::WhitenPatchesLoader>(get_build_path() + "/whiten-filters/" + dataset_name);
		experiment->push<process::SeparateSign>();
	}
	else
	{
		if (config["to_grayscale"] == true)
		{
			experiment->push<process::GrayScale>();
		}
		if (!config["dog"].isNull())
		{
			experiment->push<process::DefaultOnOffFilter>(config["dog"][0], config["dog"][1], config["dog"][2]);
		}
	}
	if (config["feature_scaling"] == true)
	{
		experiment->push<process::FeatureScaling>();
	}
	experiment->push<LatencyCoding>();

	// Convolutional layer
	int conv1_k_w;
	int conv1_k_h;
	if (config.containsKey("conv1_k"))
	{
		conv1_k_w = config["conv1_k"];
		conv1_k_h = config["conv1_k"];
	}
	else
	{
		conv1_k_w = config["conv1_k_w"];
		conv1_k_h = config["conv1_k_h"];
	}
	auto &conv1 = experiment->push<layer::Convolution>(conv1_k_w, conv1_k_h, config["conv1_c"]);
	conv1.set_name("conv1");
	conv1.parameter<uint32_t>("epoch").set(config["conv1_epochs"]);
	conv1.parameter<float>("annealing").set(config["conv1_annealing"]);
	conv1.parameter<float>("min_th").set(config["conv1_min_th"]);
	conv1.parameter<float>("t_obj").set(config["conv1_t_obj"]);
	conv1.parameter<float>("lr_th").set(config["conv1_lr_th"]);
	conv1.parameter<bool>("wta_infer").set(config["conv1_wta_infer"]);
	conv1.parameter<Tensor<float>>("w").distribution<distribution::Gaussian>(config["conv1_w_init_mean"], config["conv1_w_init_std"]);
	conv1.parameter<Tensor<float>>("th").distribution<distribution::Constant>(config["conv1_th"].as<float>());
	std::string stdp_type(config["conv1_stdp"].as<const char *>());
	float ap;
	float am;
	if (config["conv1_stdp_lr"].is<JsonArray>())
	{
		ap = config["conv1_stdp_lr"][0];
		am = config["conv1_stdp_lr"][1];
	}
	else
	{
		ap = config["conv1_stdp_lr"];
		am = config["conv1_stdp_lr"];
	}
	if (stdp_type == "multiplicative")
	{
		conv1.parameter<STDP>("stdp").set<stdp::Multiplicative>(ap, am, config["conv1_stdp_b"]);
	}
	else if (stdp_type == "biological")
	{
		conv1.parameter<STDP>("stdp").set<stdp::Biological>(ap, am, config["conv1_stdp_t"]);
	}
	else
	{
		throw std::runtime_error("STDP type " + stdp_type + " is not implemented");
	}

	// Pooling layer
	auto &pool1 = experiment->push<layer::Pooling>(config["pool1_size"], config["pool1_size"], config["pool1_size"], config["pool1_size"]);
	pool1.set_name("pool1");

	// Convolutional layer
	int conv2_k_w;
	int conv2_k_h;
	if (config.containsKey("conv2_k"))
	{
		conv2_k_w = config["conv2_k"];
		conv2_k_h = config["conv2_k"];
	}
	else
	{
		conv2_k_w = config["conv2_k_w"];
		conv2_k_h = config["conv2_k_h"];
	}
	auto &conv2 = experiment->push<layer::Convolution>(conv2_k_w, conv2_k_h, config["conv2_c"]);
	conv2.set_name("conv2");
	conv2.parameter<uint32_t>("epoch").set(config["conv2_epochs"]);
	conv2.parameter<float>("annealing").set(config["conv2_annealing"]);
	conv2.parameter<float>("min_th").set(config["conv2_min_th"]);
	conv2.parameter<float>("t_obj").set(config["conv2_t_obj"]);
	conv2.parameter<float>("lr_th").set(config["conv2_lr_th"]);
	conv2.parameter<bool>("wta_infer").set(config["conv2_wta_infer"]);
	conv2.parameter<Tensor<float>>("w").distribution<distribution::Gaussian>(config["conv2_w_init_mean"], config["conv2_w_init_std"]);
	conv2.parameter<Tensor<float>>("th").distribution<distribution::Constant>(config["conv2_th"].as<float>());
	std::string stdp_type2(config["conv2_stdp"].as<const char *>());
	if (config["conv2_stdp_lr"].is<JsonArray>())
	{
		ap = config["conv2_stdp_lr"][0];
		am = config["conv2_stdp_lr"][1];
	}
	else
	{
		ap = config["conv2_stdp_lr"];
		am = config["conv2_stdp_lr"];
	}
	if (stdp_type2 == "multiplicative")
	{
		conv2.parameter<STDP>("stdp").set<stdp::Multiplicative>(ap, am, config["conv2_stdp_b"]);
	}
	else if (stdp_type2 == "biological")
	{
		conv2.parameter<STDP>("stdp").set<stdp::Biological>(ap, am, config["conv2_stdp_t"]);
	}
	else
	{
		throw std::runtime_error("STDP type " + stdp_type2 + " is not implemented");
	}

	// Pooling layer
	auto &pool2 = experiment->push<layer::Pooling>(config["pool2_size"], config["pool2_size"], config["pool2_size"], config["pool2_size"]);
	pool2.set_name("pool2");

	// Activity analysis
	auto &conv1_activity = experiment->output<DefaultOutput>(conv1, 0.0, 1.0);
	conv1_activity.add_analysis<analysis::Coherence>();
	auto &pool1_activity = experiment->output<DefaultOutput>(pool1, 0.0, 1.0);
	pool1_activity.add_analysis<analysis::Activity>();
	auto &conv2_activity = experiment->output<DefaultOutput>(conv2, 0.0, 1.0);
	conv2_activity.add_analysis<analysis::Coherence>();
	auto &pool2_activity = experiment->output<DefaultOutput>(pool2, 0.0, 1.0);
	pool2_activity.add_analysis<analysis::Activity>();

	// Save feature maps
	auto &pool2_save = experiment->output<SpikeTiming>(pool2);
	pool2_save.add_analysis<analysis::SaveFeatureNumpy>(output_path);

	experiment->run(10000);
	delete experiment;
}
