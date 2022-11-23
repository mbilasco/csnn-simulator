#include "Benchmark.h"
#include "process/OnOffFilter.h"
#include "process/Scaling.h"
#include "process/Pooling.h"
#include "layer/Convolution.h"
#include "layer/Pooling.h"
#include "stdp/Linear.h"
#include "analysis/Activity.h"
#include "analysis/Coherence.h"
#include "analysis/Svm.h"
#include "dataset/FaceMotor.h"
#include "execution/DenseIntermediateExecution.h"
#include "stdp/BiologicalMultiplicative.h"

#ifdef ENABLE_QT
template<typename T>
struct Handler {

	static void run(Experiment<T>& experiment, const std::map<std::string, float>& variables) {
/*
		experiment.template add_preprocessing<process::DefaultOnOffFilter>(7, 1.0, 4.0);
		experiment.template add_preprocessing<process::FeatureScaling>();
		experiment.template input<LatencyCoding>();

		const char* input_path_ptr = std::getenv("INPUT_PATH");

		if(input_path_ptr == nullptr) {
			throw std::runtime_error("Require to define INPUT_PATH variable");
		}

		std::string input_path(input_path_ptr);


		experiment.template add_train<dataset::FaceMotor>(input_path+"TrainingSet/Face", input_path+"TrainingSet/Motor");
		experiment.template add_test<dataset::FaceMotor>(input_path+"TestingSet/Face", input_path+"TestingSet/Motor");

		float t_obj = variables.at("t_obj");
		float lr = 0.1f;

		auto& conv1 = experiment.template push_layer<layer::Convolution>("conv1", 5, 5, 32, 1, 1, 5/2, 5/2);
		conv1.template parameter<float>("annealing").set(0.95f);
		conv1.template parameter<float>("min_th").set(8.0f);
		conv1.template parameter<float>("t_obj").set(t_obj);
		conv1.template parameter<float>("lr_th").set(lr*10.0f);
		conv1.template parameter<Tensor<float>>("w").template distribution<distribution::Gaussian>(0.8, 0.01);
		conv1.template parameter<Tensor<float>>("th").template distribution<distribution::Gaussian>(5.0, 1.0);
		conv1.template parameter<STDP>("stdp").template set<stdp::Linear>(lr, lr);

		experiment.template push_layer<layer::Pooling>("pool1", 7, 7, 6, 6, 7/2, 7/2);

		auto& conv2 = experiment.template push_layer<layer::Convolution>("conv2", 17, 17, 64, 1, 1, 17/2, 17/2);
		conv2.template parameter<float>("annealing").set(0.95f);
		conv2.template parameter<float>("min_th").set(1.0f);
		conv2.template parameter<float>("t_obj").set(t_obj);
		conv2.template parameter<float>("lr_th").set(lr*10.0f);
		conv2.template parameter<Tensor<float>>("w").template distribution<distribution::Uniform>(0.0, 1.0);
		conv2.template parameter<Tensor<float>>("th").template distribution<distribution::Gaussian>(5.0, 1.0);
		conv2.template parameter<STDP>("stdp").template set<stdp::Linear>(lr, lr);

		experiment.template push_layer<layer::Pooling>("pool2", 5, 5, 5, 5, 5/2, 5/2);

		// TODO fully conencted
		auto& fc1 = experiment.template push_layer<layer::Convolution>("fc1", 5, 5, 128, 1, 1, 5/2, 5/2);
		fc1.template parameter<float>("annealing").set(0.95f);
		fc1.template parameter<float>("min_th").set(1.0f);
		fc1.template parameter<float>("t_obj").set(t_obj);
		fc1.template parameter<float>("lr_th").set(lr*10.0f);
		fc1.template parameter<Tensor<float>>("w").template distribution<distribution::Uniform>(0.0, 1.0);
		fc1.template parameter<Tensor<float>>("th").template distribution<distribution::Gaussian>(5.0, 1.0);
		fc1.template parameter<STDP>("stdp").template set<stdp::Linear>(lr, lr);

		experiment.add_train_step(conv1, 100);
		experiment.add_train_step(conv2, 100);
		experiment.add_train_step(fc1, 100);

		auto& conv1_out = experiment.template output<TimeObjectiveOutput>(conv1, t_obj);
		conv1_out.template add_postprocessing<process::SumPooling>(1, 1);
		conv1_out.template add_postprocessing<process::FeatureScaling>();
		conv1_out.template add_analysis<analysis::Activity>();
		conv1_out.template add_analysis<analysis::Coherence>();
		conv1_out.template add_analysis<analysis::Svm>();

		auto& conv2_out = experiment.template output<TimeObjectiveOutput>(conv2, t_obj);
		conv2_out.template add_postprocessing<process::SumPooling>(1, 1);
		conv2_out.template add_postprocessing<process::FeatureScaling>();
		conv2_out.template add_analysis<analysis::Activity>();
		conv2_out.template add_analysis<analysis::Coherence>();
		conv2_out.template add_analysis<analysis::Svm>();

		auto& fc1_out = experiment.template output<TimeObjectiveOutput>(fc1, t_obj);
		fc1_out.template add_postprocessing<process::SumPooling>(1, 1);
		fc1_out.template add_postprocessing<process::FeatureScaling>();
		fc1_out.template add_analysis<analysis::Activity>();
		fc1_out.template add_analysis<analysis::Coherence>();
		fc1_out.template add_analysis<analysis::Svm>();
*/
	}
};
#endif

int main(int argc, char** argv) {
	/*
#ifdef ENABLE_QT
	Benchmark<Handler, OptimizedLayerByLayer> bench(argc, argv, "bench-face_motor-1", 10);
	bench.add_variable("t_obj", {0.2f, 0.3f, 0.4f,  0.5f, 0.6f, 0.7f, 0.8f, 0.9f});

	bench.run(4);
#endif*/
}
