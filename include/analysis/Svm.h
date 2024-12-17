#ifndef _ANALYSIS_SVM_H
#define _ANALYSIS_SVM_H

#include "Analysis.h"
#include <filesystem>
#include "tool/Operations.h"

	/**
 	* @brief SVM (support vector machine), this CSNN simulator uses STDP unsupervised learning, 
	* so a classification layer is needed to evaluate the accuracy of the network.
	* Any other supervised learning method can be used, but the SVM was chosen for it's simplicity and efficacity.
    * @param draw A flag that draws the features that will be classified by the SVM, the information is recorded in the folder in the build file.
 	*/

namespace analysis {
	class Svm : public TwoPassAnalysis {

	public:
		Svm();
		Svm(const size_t &draw);
		~Svm();

		Svm(const Svm& that) = delete;
		Svm& operator=(const Svm& that) = delete;

		virtual void resize(const Shape& shape);
		virtual void compute(const std::string& label, const Tensor<float>& sample);
		virtual void process_train(const std::string& label, const Tensor<float>& sample);
		virtual void process_test(const std::string& label, const Tensor<float>& sample);

		virtual void before_train();
		virtual void after_train();
		virtual void before_test();
		virtual void after_test();

	private:
		void process_as_npy_sample(const std::string& label, const Tensor<float>& sample, std::vector<int> &_labels_npy, std::vector<float> &_samples_npy);
		void call_python_binding(const std::vector<int> &train_labels, const std::vector<float> &train_samples, const std::vector<int> &test_labels, const std::vector<float> &test_samples);
		float _c;

		std::map<std::string, double> _label_index;
		std::vector<int> _train_labels,_test_labels;
		std::vector<float> _train_samples,_test_samples;
		
		size_t _sample_count;
		size_t _draw;

		size_t _correct_sample;
		size_t _total_sample;

		size_t _correct_sample_train;
		size_t _total_sample_train;
		
		//size_t _tensor_size;
	};
}

#endif
