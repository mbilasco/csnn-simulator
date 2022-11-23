#include "Experiment.h"
#include "dataset/STL.h"
#include "stdp/Multiplicative.h"
#include "stdp/Biological.h"
#include "layer/Convolution.h"
#include "layer/Pooling.h"
#include "Distribution.h"
#include "execution/DenseIntermediateExecution.h"
#include "analysis/Svm.h"
#include "analysis/Activity.h"
#include "analysis/Coherence.h"
#include "process/Input.h"
#include "process/Scaling.h"
#include "process/Pooling.h"
#include "process/OnOffFilter.h"

int main(int argc, char** argv) {
/*
	Experiment<OptimizedLayerByLayer> experiment(argc, argv, "stl_10_1");

	const char* input_path_ptr = std::getenv("INPUT_PATH");

	if(input_path_ptr == nullptr) {
		throw std::runtime_error("Require to define INPUT_PATH variable");
	}

	std::string input_path(input_path_ptr);

	experiment.add_preprocessing<process::BiologicalOnOffFilter>(7, 1.0, 4.0);
	experiment.add_preprocessing<process::FeatureScaling>();
	experiment.input<LatencyCoding>();

	experiment.add_train<dataset::STL>(input_path+"train_X.bin", input_path+"train_y.bin");
	experiment.add_test<dataset::STL>(input_path+"test_X.bin", input_path+"test_y.bin");

	float t_obj = 0.85f;
	float th_lr = 1.0f;
	float w_lr = 0.1f;

	auto& conv1 = experiment.push_layer<layer::Convolution>("conv1", 5, 5, 128);
	conv1.parameter<float>("annealing").set(0.95f);
	conv1.parameter<float>("min_th").set(4.0f);
	conv1.parameter<float>("t_obj").set(t_obj);
	conv1.parameter<float>("lr_th").set(th_lr);
	conv1.parameter<Tensor<float>>("w").distribution<distribution::Uniform>(0.0, 1.0);
	conv1.parameter<Tensor<float>>("th").distribution<distribution::Gaussian>(5.0, 0.1);
	conv1.parameter<STDP>("stdp").set<stdp::Biological>(w_lr, 0.1);

	experiment.add_train_step(conv1, 100);

	auto& conv1_out = experiment.output<TimeObjectiveOutput>(conv1, t_obj);
	conv1_out.add_postprocessing<process::SumPooling>(2, 2);
	conv1_out.add_postprocessing<process::FeatureScaling>();
	conv1_out.add_analysis<analysis::Activity>();
	conv1_out.add_analysis<analysis::Coherence>();
	conv1_out.add_analysis<analysis::Svm>();

#ifdef ENABLE_QT
	conv1.plot_threshold(true);
	conv1.plot_reconstruction<process::GaussianTemporalCodingColor>(true);
#endif

	experiment.run(10000);*/
}
