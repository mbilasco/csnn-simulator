#include "analysis/Svm.h"
#include "dep/npy.hpp"
#include "Experiment.h"
#include <Python.h>

using namespace analysis;

static RegisterClassParameter<Svm, AnalysisFactory> _register("Svm");

Svm::Svm() : TwoPassAnalysis(_register),
			 _c(0), _label_index(), _size(0), _node_count(0), _sample_count(0),
			 _problem(), _model(nullptr), _train_nodes(nullptr), _test_nodes(nullptr),
			 _correct_sample(0), _total_sample(0)
{

	add_parameter("c", _c, 1.0f);

	_problem.l = 0;
	_problem.x = nullptr;
	_problem.y = nullptr;
}

Svm::Svm(const size_t &draw) : TwoPassAnalysis(_register),
							   _draw(draw), _c(0), _label_index(), _size(0), _sample_count(0),
							   _correct_sample(0), _total_sample(0)
{
	add_parameter("c", _c, 1.0f);

}

void Svm::resize(const Shape& shape) {
	_node_count = 0;
	_sample_count = 0;
	_size = shape.product();
	_label_index.clear();
}

void Svm::compute(const std::string& label, const Tensor<float>& sample) {

}

void Svm::before_train() {
	_train_labels.clear();
	_train_samples.clear();
	_sample_count = 0;
	_node_count = 0;

}

void Svm::process_train(const std::string& label, const Tensor<float>& sample) {
	
	_train_labels.emplace_back(label);
	_train_samples.emplace_back(sample);
	_sample_count++;
}

void Svm::after_train() {

	const bool fortran_order{false};
    
    const std::vector<long unsigned> shape_data{_sample_count, _train_samples.at(0).size()};
    const std::vector<long unsigned> shape_label{_sample_count};

    npy::SaveArrayAsNumpy(_path + '/' + "X_train_for_SVM.npy", fortran_order, shape_data.size(), shape_data.data(), _data_train);
    npy::SaveArrayAsNumpy(_path + '/' + "X_train_for_SVM.npy", fortran_order, shape_label.size(), shape_label.data(), _label_train);
	
	/*
	parameters.svm_type = C_SVC;
	parameters.kernel_type = LINEAR;
	parameters.degree = 3;
	parameters.gamma = 1.0/static_cast<float>(_size);
	parameters.coef0 = 0;
	parameters.nu = 0.5;
	parameters.cache_size = 100;
	parameters.C = _c;
	parameters.eps = 1e-3;
	parameters.p = 0.1;
	parameters.shrinking = 1;
	parameters.probability = 0;
	parameters.nr_weight = 0;
	parameters.weight_label = NULL;
	parameters.weight = NULL;
	*/

	std::fstream python_script;
	python_script.open("svm_sklearn.py", std::fstream::out | std::fstream::app);
	python_script<<"import numpy as np\n"
	python_script<<"import sklearn.svm as skm\n";
	python_script<<"def linearSVM():"
	python_script<<"    data,labels=np.load('"<<_path<<"/X_train_for_SVM.npy'),np.load('"<<_path<<"/y_train_for_SVM.npy')\n";
	python_script<<"    svm=skm.LinearSVC()\n";
	python_script<<"    svm.fit(data,labels)\n";
	python_script<<"    score=svm.score(data,labels)\n";
	python_script<<"    np.save('"<<_path<<"/SVM_score.npy')\n";
	
	PyObject *pName, *pModule, *pDict, *pFunc, *pValue;
	// Initialize the Python Interpreter
    Py_Initialize();
    // Build the name object
    pName = PyString_FromString("svm_sklearn.py");
    // Load the module object
    pModule = PyImport_Import(pName);
	pDict = PyModule_GetDict(pModule);
    // pFunc is also a borrowed reference 
    pFunc = PyDict_GetItemString(pDict,"linearSVM");
    if (PyCallable_Check(pFunc)) 
    {
        PyObject_CallObject(pFunc, NULL);
    } else 
    {
        PyErr_Print();
    }

    // Clean up
    Py_DECREF(pModule);
    Py_DECREF(pName);

    // Finish the Python Interpreter
    Py_Finalize();

	const std::vector<long unsigned> score_shape_data{1};

	std::vector<double> _score;

    npy::LoadArrayFromNumpy(_path + '/' + "SVM_score.npy", fortran_order, score_shape_data.size(), _score);
    
	experiment().print() << "Train svm" << std::endl;
	experiment().print() << "Score : " << _score.at(0) << std::endl;
	
}

void Svm::before_test() {
	_test_labels.clear();
	_test_samples.clear();
	_correct_sample = 0;
	_total_sample = 0;

}

void Svm::process_test(const std::string& label, const Tensor<float>& sample) {
	_test_labels.emplace_back(label);
	_test_samples.emplace_back(sample);
	_sample_count++;
}

void Svm::after_test() {
	//count correct
	if(it != std::end(_label_index) && y_pred == it->second) {
		_correct_sample++;
	}
	_total_sample++;

	experiment().log() << "===SVM===" << std::endl;
	experiment().log() << "classification rate: " <<
						 (static_cast<float>(_correct_sample)/static_cast<float>(_total_sample)*100.0) << "% (" <<
						 _correct_sample << "/" << _total_sample << ")" << std::endl;
	experiment().log() << std::endl;

	delete[] _problem.y;
	_problem.y = nullptr;
	delete[] _problem.x;
	_problem.x = nullptr;
	delete[] _train_nodes;
	_train_nodes = nullptr;
	delete[] _test_nodes;
	_test_nodes = nullptr;

	svm_free_and_destroy_model(&_model);
	_model = nullptr;
}
