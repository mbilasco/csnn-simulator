#include "Experiment.h"
#include "dataset/Cifar.h"
#include "dataset/Cifar10.h"
#include "dataset/Mnist.h"
#include "dataset/MnistCustom.h"
#include "dataset/ETH.h"
#include "dataset/STL.h"
#include "stdp/Multiplicative.h"
#include "stdp/Biological.h"
#include "layer/Convolution.h"
#include "layer/Pooling.h"
#include "Distribution.h"
#include "execution/TestingExecution.h"
#include "execution/TestingSparseExecution.h"
#include "analysis/SaveLayer.h"
#include "analysis/SaveOutputNumpy.h"
#include "analysis/Activity.h"
#include "analysis/Coherence.h"
#include "process/Input.h"
#include "process/Scaling.h"
#include "process/Pooling.h"
#include "process/GrayScale.h"
#include "process/OnOffFilter.h"
#include "dep/filesystem.hpp"
#include "dep/ArduinoJson-v6.17.3.h"


void load_dataset(AbstractExperiment* experiment, std::string& data_path, std::string& label_path, std::string& dataset) {
    if (dataset == "MNIST") {
		experiment->add_test<dataset::MnistCustom>(data_path, label_path);
    }
    else if (dataset == "CIFAR10") {
		experiment->add_test<dataset::Cifar10>(data_path, label_path);
	}
    else if (dataset == "STL10") {
        experiment->add_test<dataset::STL>(data_path, label_path);
    }
    else if (dataset == "ETH80") {
		experiment->add_test<dataset::ETH>(data_path, label_path);
    }
    else {
        throw std::runtime_error("Dataset loader for " + dataset + " is not implemented");
    }
}


int main(int argc, char** argv) {

	// Argument parsing
    if (argc < 5 || argc > 6) {
		throw std::runtime_error("Usage: " + std::string(argv[0]) + " <DATA_PATH> <LABEL_PATH> <CONFIG_PATH> <MODEL_PATH> [ <OUTPUT_PATH = ./> ]");
    }

    std::string data_path = std::string(argv[1]);
	std::string label_path = std::string(argv[2]);
	std::string config_path = std::string(argv[3]);
	std::string model_path = std::string(argv[4]);
	std::string output_path = "./";
	if (argc > 5) {
		output_path = std::string(argv[5]);
	}
	ghc::filesystem::create_directories(output_path);

	// Load config
	std::ifstream _jsonTextFile(config_path);
	if (!_jsonTextFile.good()) {
		throw std::runtime_error("Failed to open JSON config");
	}
	std::stringstream buffer;
	buffer << _jsonTextFile.rdbuf();
	std::string _jsonText = buffer.str();
	_jsonTextFile.close();
	DynamicJsonDocument config(JSON_ARRAY_SIZE(_jsonText.length()));
	DeserializationError error = deserializeJson(config, _jsonText.c_str());
	if (error) {
		throw std::runtime_error("Failed to parse JSON config");
	}

	// Initialize experiment
	AbstractExperiment* experiment;
	std::string exp_name = "test";
	if (config["use_sparse"] == true) {
		experiment = new Experiment<TestingExecution>(argc, argv, output_path, exp_name, 0, false);
	}
	else {
		experiment = new Experiment<TestingSparseExecution>(argc, argv, output_path, exp_name, 0, false);
	}

	// Load dataset
    std::string dataset_name(config["dataset"].as<const char*>());
	load_dataset(experiment, data_path, label_path, dataset_name);

	// Preprocessing
	if (!config.containsKey("to_grayscale") || config["to_grayscale"] == true) {
		experiment->push<process::GrayScale>();
	}
	if (!config["dog"].isNull()) {
		experiment->push<process::DefaultOnOffFilter>(config["dog"][0], config["dog"][1], config["dog"][2]);
	}
	//if (!config.containsKey("feature_scaling") || config["feature_scaling"] == true) {
	//	experiment->push<process::FeatureScaling>();
	//}
	experiment->push<LatencyCoding>();

	// Convolutional layer
	int conv1_k_w;
	int conv1_k_h;
	if (config.containsKey("conv1_k")) {
		conv1_k_w = config["conv1_k"];
		conv1_k_h = config["conv1_k"];
	}
	else {
		conv1_k_w = config["conv1_k_w"];
		conv1_k_h = config["conv1_k_h"];
	}
	auto& conv1 = experiment->push<layer::Convolution>(conv1_k_w, conv1_k_h, config["conv1_c"]);
	conv1.set_name("conv1");
	conv1.use_trained_model(model_path);
	conv1.parameter<uint32_t>("epoch").set(0);
	conv1.parameter<float>("annealing").set(config["conv1_annealing"]);
	conv1.parameter<float>("min_th").set(config["conv1_min_th"]);
	conv1.parameter<float>("t_obj").set(config["conv1_t_obj"]);
	conv1.parameter<float>("lr_th").set(config["conv1_lr_th"]);
	conv1.parameter<bool>("wta_infer").set(config["conv1_wta_infer"]);
	conv1.parameter<Tensor<float>>("w").distribution<distribution::Gaussian>(config["conv1_w_init_mean"], config["conv1_w_init_std"]);
	conv1.parameter<Tensor<float>>("th").distribution<distribution::Constant>(config["conv1_th"].as<float>());
	std::string stdp_type(config["conv1_stdp"].as<const char*>());
	float ap;
	float am;
	if (config["conv1_stdp_lr"].is<JsonArray>()) {
		ap = config["conv1_stdp_lr"][0];
		am = config["conv1_stdp_lr"][1];
	}
	else {
		ap = config["conv1_stdp_lr"];
		am = config["conv1_stdp_lr"];		
	}
	if (stdp_type == "multiplicative") {
		conv1.parameter<STDP>("stdp").set<stdp::Multiplicative>(ap, am, config["conv1_stdp_b"]);
	}
	else if (stdp_type == "biological") {
		conv1.parameter<STDP>("stdp").set<stdp::Biological>(ap, am, config["conv1_stdp_t"]);
	}
	else {
		throw std::runtime_error("STDP type " + stdp_type + " is not implemented");
	}

	// Pooling layer
	auto& pool1 = experiment->push<layer::Pooling>(config["pool1_size"], config["pool1_size"], config["pool1_size"], config["pool1_size"]);
	pool1.set_name("pool1");

	// Activity analysis
	auto& pool1_activity = experiment->output<DefaultOutput>(pool1, 0.0, 1.0);
	pool1_activity.add_analysis<analysis::Activity>();
	
	// Save feature maps
	auto& pool1_save = experiment->output<SpikeTiming>(pool1);
	pool1_save.add_analysis<analysis::SaveOutputNumpy>(output_path);

	experiment->run(10000);
	delete experiment;
}
