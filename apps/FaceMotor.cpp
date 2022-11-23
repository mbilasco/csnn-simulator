#include "Experiment.h"
#include "dataset/FaceMotor.h"
#include "stdp/Multiplicative.h"
#include "stdp/Biological.h"
#include "stdp/Proportional.h"
#include "layer/Convolution.h"
#include "Distribution.h"
#include "execution/DenseIntermediateExecution.h"
#include "analysis/Svm.h"
#include "analysis/Activity.h"
#include "analysis/Coherence.h"
#include "layer/Pooling.h"
#include "process/OnOffFilter.h"
#include "process/Scaling.h"
#include "process/Pooling.h"
#include "stdp/Linear.h"
#include "stdp/BiologicalMultiplicative.h"
#include "analysis/SaveOutput.h"

int main(int argc, char** argv) {
	/*
#ifdef ENABLE_QT
	Experiment<OptimizedLayerByLayer> experiment(argc, argv, "face_motor");

	experiment.add_preprocessing<process::DefaultOnOffFilter>(7, 1.0, 4.0);

	experiment.add_preprocessing<process::FeatureScaling>();
	auto& input = experiment.input<LatencyCoding>();

	const char* input_path_ptr = std::getenv("INPUT_PATH");

	if(input_path_ptr == nullptr) {
		throw std::runtime_error("Require to define INPUT_PATH variable");
	}

	std::string input_path(input_path_ptr);

	experiment.add_train<dataset::FaceMotor>(input_path+"TrainingSet/Face", input_path+"TrainingSet/Motor");
	experiment.add_test<dataset::FaceMotor>(input_path+"TestingSet/Face", input_path+"TestingSet/Motor");

	float t_obj = 0.80f;
	float lr = 0.1f;

	auto& conv1 = experiment.push_layer<layer::Convolution>("conv1", 5, 5, 32, 1, 1, 5/2, 5/2);
	conv1.parameter<float>("annealing").set(0.95f);
	conv1.parameter<float>("min_th").set(8.0f);
	conv1.parameter<float>("t_obj").set(t_obj);
	conv1.parameter<float>("lr_th").set(lr*10.0f);
	conv1.parameter<Tensor<float>>("w").distribution<distribution::Gaussian>(0.8, 0.01);
	conv1.parameter<Tensor<float>>("th").distribution<distribution::Gaussian>(5.0, 1.0);
	conv1.parameter<STDP>("stdp").set<stdp::Multiplicative>(lr, 0.0);

	experiment.push_layer<layer::Pooling>("pool1", 7, 7, 6, 6, 7/2, 7/2);

	auto& conv2 = experiment.push_layer<layer::Convolution>("conv2", 17, 17, 64, 1, 1, 17/2, 17/2);
	conv2.parameter<float>("annealing").set(0.95f);
	conv2.parameter<float>("min_th").set(1.0f);
	conv2.parameter<float>("t_obj").set(t_obj);
	conv2.parameter<float>("lr_th").set(lr*10.0f);
	conv2.parameter<Tensor<float>>("w").distribution<distribution::Uniform>(0.0, 1.0);
	conv2.parameter<Tensor<float>>("th").distribution<distribution::Gaussian>(5.0, 1.0);
	conv2.parameter<STDP>("stdp").set<stdp::Multiplicative>(lr, 0.0);

	experiment.push_layer<layer::Pooling>("pool2", 5, 5, 5, 5, 5/2, 5/2);

	auto& fc1 = experiment.push_layer<layer::Convolution>("fc1", 5, 5, 128, 1, 1, 5/2, 5/2);
	fc1.parameter<float>("annealing").set(0.95f);
	fc1.parameter<float>("min_th").set(1.0f);
	fc1.parameter<float>("t_obj").set(t_obj);
	fc1.parameter<float>("lr_th").set(lr*10.0f);
	fc1.parameter<Tensor<float>>("w").distribution<distribution::Uniform>(0.0, 1.0);
	fc1.parameter<Tensor<float>>("th").distribution<distribution::Gaussian>(5.0, 1.0);
	fc1.parameter<STDP>("stdp").set<stdp::Multiplicative>(lr, 0.0);

	experiment.add_train_step(conv1, 100);
	experiment.add_train_step(conv2, 100);
	experiment.add_train_step(fc1, 100);

	input.plot_time(false);

	conv1.plot_threshold(true);
	conv1.plot_reconstruction(true);
	conv1.plot_time(false);

	conv2.plot_threshold(true);
	conv2.plot_reconstruction(true, 20);
	conv2.plot_time(false);

	fc1.plot_threshold(true);
	fc1.plot_reconstruction(true, 10);
	fc1.plot_time(false);

	auto& conv1_out = experiment.output<TimeObjectiveOutput>(conv1, t_obj);
	conv1_out.add_postprocessing<process::SumPooling>(1, 1);
	conv1_out.add_postprocessing<process::FeatureScaling>();
	conv1_out.add_analysis<analysis::Activity>();
	conv1_out.add_analysis<analysis::Coherence>();
	conv1_out.add_analysis<analysis::Svm>();

	auto& conv2_out = experiment.output<TimeObjectiveOutput>(conv2, t_obj);
	conv2_out.add_postprocessing<process::SumPooling>(1, 1);
	conv2_out.add_postprocessing<process::FeatureScaling>();
	conv2_out.add_analysis<analysis::Activity>();
	conv2_out.add_analysis<analysis::Coherence>();
	conv2_out.add_analysis<analysis::Svm>();

	auto& fc1_out = experiment.output<TimeObjectiveOutput>(fc1, t_obj);
	fc1_out.add_postprocessing<process::SumPooling>(1, 1);
	fc1_out.add_postprocessing<process::FeatureScaling>();
	fc1_out.add_analysis<analysis::Activity>();
	fc1_out.add_analysis<analysis::Coherence>();
	fc1_out.add_analysis<analysis::Svm>();

	experiment.run(100);
#endif*/
}
