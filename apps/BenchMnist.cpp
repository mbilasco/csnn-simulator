#include "Benchmark.h"
#include "process/OnOffFilter.h"
#include "process/Scaling.h"
#include "process/Pooling.h"
#include "layer/Convolution.h"
#include "layer/Pooling.h"
#include "stdp/Biological.h"
#include "analysis/Activity.h"
#include "analysis/Coherence.h"
#include "analysis/Svm.h"
#include "dataset/Mnist.h"
#include "execution/DenseIntermediateExecution.h"
#include "stdp/BiologicalMultiplicative.h"

template<typename T>
struct Handler {

	static void run(Experiment<T>& experiment, const std::map<std::string, float>& variables) {/*
		experiment.template add_preprocessing<process::DefaultOnOffFilter>(7, 1.0, 4.0);
		experiment.template add_preprocessing<process::FeatureScaling>();
		experiment.template input<LatencyCoding>();

		const char* input_path_ptr = std::getenv("INPUT_PATH");

		if(input_path_ptr == nullptr) {
			throw std::runtime_error("Require to define INPUT_PATH variable");
		}

		std::string input_path(input_path_ptr);

		experiment.template add_train<dataset::Mnist>(input_path+"train-images.idx3-ubyte", input_path+"train-labels.idx1-ubyte");
		experiment.template add_test<dataset::Mnist>(input_path+"t10k-images.idx3-ubyte", input_path+"t10k-labels.idx1-ubyte");

		float w_lr = 0.1f;
		float th_lr = 1.0f;

		float t_obj = variables.at("t_obj");

		auto& conv1 = experiment.template push_layer<layer::Convolution>("conv1", 5, 5, 32);
		conv1.template parameter<float>("annealing").set(0.95f);
		conv1.template parameter<float>("min_th").set(1.0f);
		conv1.template parameter<float>("t_obj").set(t_obj);
		conv1.template parameter<float>("lr_th").set(th_lr);
		conv1.template parameter<Tensor<float>>("w").template distribution<distribution::Uniform>(0.0, 1.0);
		conv1.template parameter<Tensor<float>>("th").template distribution<distribution::Gaussian>(8.0, 0.1);
		conv1.template parameter<STDP>("stdp").template set<stdp::Biological>(w_lr, 0.1f);

		experiment.template push_layer<layer::Pooling>("pool1", 2, 2, 2, 2);

		auto& conv2 = experiment.template push_layer<layer::Convolution>("conv2", 5, 5, 128);
		conv2.template parameter<float>("annealing").set(0.95f);
		conv2.template parameter<float>("min_th").set(1.0f);
		conv2.template parameter<float>("t_obj").set(t_obj);
		conv2.template parameter<float>("lr_th").set(th_lr);
		conv2.template parameter<Tensor<float>>("w").template distribution<distribution::Uniform>(0.0, 1.0);
		conv2.template parameter<Tensor<float>>("th").template distribution<distribution::Gaussian>(10.0, 0.1);
		conv2.template parameter<STDP>("stdp").template set<stdp::Biological>(w_lr, 0.1f);

		experiment.template push_layer<layer::Pooling>("pool2", 2, 2, 2, 2);

		auto& fc1 = experiment.template push_layer<layer::Convolution>("fc1", 4, 4, 4096);
		fc1.template parameter<float>("annealing").set(0.95f);
		fc1.template parameter<float>("min_th").set(1.0f);
		fc1.template parameter<float>("t_obj").set(t_obj);
		fc1.template parameter<float>("lr_th").set(th_lr);
		fc1.template parameter<Tensor<float>>("w").template distribution<distribution::Uniform>(0.0, 1.0);
		fc1.template parameter<Tensor<float>>("th").template distribution<distribution::Gaussian>(10.0, 0.1);
		fc1.template parameter<STDP>("stdp").template set<stdp::Biological>(w_lr, 0.1f);

		experiment.add_train_step(conv1, 100);
		experiment.add_train_step(conv2, 100);
		experiment.add_train_step(fc1, 100);

		auto& conv1_out = experiment.template output<TimeObjectiveOutput>(conv1, t_obj);
		conv1_out.template add_postprocessing<process::SumPooling>(2, 2);
		conv1_out.template add_postprocessing<process::FeatureScaling>();
		conv1_out.template add_analysis<analysis::Activity>();
		conv1_out.template add_analysis<analysis::Coherence>();
		conv1_out.template add_analysis<analysis::Svm>();

		auto& conv2_out = experiment.template output<TimeObjectiveOutput>(conv2, t_obj);
		conv2_out.template add_postprocessing<process::SumPooling>(2, 2);
		conv2_out.template add_postprocessing<process::FeatureScaling>();
		conv2_out.template add_analysis<analysis::Activity>();
		conv2_out.template add_analysis<analysis::Coherence>();
		conv2_out.template add_analysis<analysis::Svm>();

		auto& fc1_out = experiment.template output<TimeObjectiveOutput>(fc1, t_obj);
		fc1_out.template add_postprocessing<process::FeatureScaling>();
		fc1_out.template add_analysis<analysis::Activity>();
		fc1_out.template add_analysis<analysis::Coherence>();
		fc1_out.template add_analysis<analysis::Svm>();*/
	}
};

int main(int argc, char** argv) {
	/*
	Benchmark<Handler, OptimizedLayerByLayer> bench(argc, argv, "bench-mnist", 10);
	bench.add_variable("t_obj", {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f});

	bench.run(16);*/
}
