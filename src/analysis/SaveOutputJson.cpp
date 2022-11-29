#include "analysis/SaveOutputJson.h"
#include <iostream>

using namespace analysis;

static RegisterClassParameter<SaveOutputJson, AnalysisFactory> _register("SaveOutputJson");

SaveOutputJson::SaveOutputJson() : UniquePassAnalysis(_register),
	_train_filename(), _test_filename(), _json_train_file(), _json_test_file() {
	throw std::runtime_error("Unimplemented");
}

SaveOutputJson::SaveOutputJson(const std::string& train_filename, const std::string& test_filename) : UniquePassAnalysis(_register),
	_train_filename(train_filename), _test_filename(test_filename), _json_train_file(), _json_test_file() {

}

void SaveOutputJson::resize(const Shape&) {

}

void SaveOutputJson::before_train() {
	_json_train_file.open(_train_filename, std::ios_base::app);
	_json_train_file << "[";
}

void SaveOutputJson::process_train(const std::string& label, const Tensor<float>& sample) {
	// Convert samlple to JSON string
	std::string JSON_output = _to_json_string(label, sample);
	// Append to the file
	_json_train_file << JSON_output;
	_json_train_file << ",";
}

void SaveOutputJson::after_train() {
	_json_train_file << "]";
    _json_train_file.close();
	//delete _json_train_file;
	//_json_train_file = nullptr;
}

void SaveOutputJson::before_test() {
	_json_test_file.open(_test_filename, std::ios_base::app);
	_json_test_file << "[";
}

void SaveOutputJson::process_test(const std::string& label, const Tensor<float>& sample) {
	// Convert samlple to JSON string
	std::string JSON_output = _to_json_string(label, sample);
	// Append to the file
	_json_test_file << JSON_output;
	_json_test_file << ",";
}

void SaveOutputJson::after_test() {
	_json_test_file << "]";
    _json_test_file.close();
	//delete _json_test_file;
	//_json_test_file = nullptr;
}

std::string SaveOutputJson::_to_json_string(const std::string& label, const Tensor<float>& sample) {
    std::string JSON_output = "";

	std::cout << sample.at(1,1,0) << std::endl;
	// Convert tensor of spike times into vector of spikes
	std::vector<Spike> spks;
	SpikeConverter::to_spike(sample, spks);

	// Create a JSON document that holds the memory of the sample to serialize
	// 4 bytes for the label + 4 bytes * 4 (for x,y,z,time vars) * n_spks
	DynamicJsonDocument doc(JSON_OBJECT_SIZE(4 + 4 * 4 * spks.size()));

	// Create the main object
	JsonObject root = doc.to<JsonObject>();

	// Fill label info
	root["label"] = label;

	// Array containing spikes
	JsonArray spks_arr = root.createNestedArray("data");

	// Filling it with spike objects
	for(Spike spk : spks)
	{
		JsonObject spk_obj = spks_arr.createNestedObject();
		spk_obj["x"] = spk.x;
		spk_obj["y"] = spk.y;
		spk_obj["z"] = spk.z;
		spk_obj["time"] = spk.time;
	}

	// Transform to string
	serializeJson(doc, JSON_output);

	return JSON_output;
}