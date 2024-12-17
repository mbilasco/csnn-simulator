#define PY_SSIZE_T_CLEAN

#include "analysis/Svm.h"
#include "dep/npy.hpp"
#include "Experiment.h"
#include <Python.h>

using namespace analysis;

static RegisterClassParameter<Svm, AnalysisFactory> _register("Svm");

Svm::Svm() : TwoPassAnalysis(_register), _draw(0),
			 _c(0), _sample_count(0),
	_train_labels(0),_test_labels(0),_train_samples(0),_test_samples(0),			 
			 _correct_sample(0), _total_sample(0), 
			 _correct_sample_train(0), _total_sample_train(0)
{

	add_parameter("c", _c, 1.0f);
}

Svm::Svm(const size_t &draw) : TwoPassAnalysis(_register),
							   _draw(draw), _c(0), _sample_count(0),
							   _correct_sample(0), _total_sample(0),
	_train_labels(0),_test_labels(0),_train_samples(0),_test_samples(0),
			 _correct_sample_train(0), _total_sample_train(0)
{
	add_parameter("c", _c, 1.0f);

}

void Svm::resize(const Shape& shape) {
	_sample_count = 0;
	//experiment().log()<<"RESIZING WITH shape :"<<shape.number()<<" - "<<shape.product()<<"\n";
	//_tensor_size = shape.product();
}

void Svm::compute(const std::string& label, const Tensor<float>& sample) {
	/*if (_tensor_size==0) {
		//std::cerr<<"Processing shape :"<<sample.shape().number()<<" - "<<sample.shape().product()<<"\n";
		_tensor_size=sample.shape().product();
	}*/
}

void Svm::before_train() {
	_train_labels.clear();
	_train_samples.clear();
//	_tensor_size = 0;
	_sample_count = 0;
}

void Svm::process_as_npy_sample(const std::string& label, const Tensor<float>& sample, std::vector<int> &_labels_npy, std::vector<float> &_samples_npy) {

	_labels_npy.emplace_back(std::stoi(label));

	size_t _width = sample.shape().dim(0);
	size_t _height = sample.shape().dim(1);
	bool _is3D=sample.shape().number()>2;
	size_t _depth = _is3D?sample.shape().dim(2):1;
	bool _is4D=sample.shape().number()>3;
	size_t _conv_depth = _is4D?sample.shape().dim(3):1;

	for(size_t x=0; x<_width; x++) {
		for(size_t y=0; y<_height; y++) {
			if (_is3D) {
				for(size_t z=0; z<_depth; z++) {
					if (_is4D) {
						for(size_t t=0; t<_conv_depth; t++) {
                					_samples_npy.emplace_back(sample.at(x, y, z,t));
						}
					} else {
                				_samples_npy.emplace_back(sample.at(x, y, z));
					}	
				}
			} else {
                		_samples_npy.emplace_back(sample.at(x, y));
			}
            }
        }
}

void Svm::process_train(const std::string& label, const Tensor<float>& sample) {
	process_as_npy_sample(label, sample, _train_labels, _train_samples);
	_sample_count++;
}

void Svm::call_python_binding(const std::vector<int> &train_labels, const std::vector<float> &train_samples, const std::vector<int> &test_labels, const std::vector<float> &test_samples ) {
   
    std::string _path="/tmp/"+std::to_string(gettid());
   
    bool fortran_order{false};
    int descr_size=train_samples.size()/train_labels.size();

    const std::vector<long unsigned> train_shape_data{train_labels.size(), descr_size};
    const std::vector<long unsigned> train_shape_label{train_labels.size()};
    npy::SaveArrayAsNumpy(_path + '_' + "X_train_for_SVM.npy", fortran_order, train_shape_data.size(), train_shape_data.data(), train_samples);
    npy::SaveArrayAsNumpy(_path + '_' + "y_train_for_SVM.npy", fortran_order, train_shape_label.size(), train_shape_label.data(), train_labels);
	
    const std::vector<long unsigned> test_shape_data{test_labels.size(), descr_size};
    const std::vector<long unsigned> test_shape_label{test_labels.size()};
    npy::SaveArrayAsNumpy(_path + '_' + "X_test_for_SVM.npy", fortran_order, test_shape_data.size(), test_shape_data.data(), test_samples);
    npy::SaveArrayAsNumpy(_path + '_' + "y_test_for_SVM.npy", fortran_order, test_shape_label.size(), test_shape_label.data(), test_labels);
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
	python_script.open(_path+"_svm_sklearn.py", std::fstream::out);
	python_script<<"import numpy as np\n";
	python_script<<"import sklearn.svm as skm\n";
	python_script<<"def linearSVM():\n";
	python_script<<"\tdata,labels=np.load('"<<_path<<"_X_train_for_SVM.npy'),np.load('"<<_path<<"_y_train_for_SVM.npy')\n";
	python_script<<"\ttest_data,test_labels=np.load('"<<_path<<"_X_test_for_SVM.npy'),np.load('"<<_path<<"_y_test_for_SVM.npy')\n";
	python_script<<"\tprint('data loaded')\n";
	python_script<<"\tsvm=skm.LinearSVC()\n";
	python_script<<"\tsvm.fit(data,labels)\n";
	python_script<<"\ttrain_score=svm.score(data,labels)\n";
	python_script<<"\ttest_score=svm.score(test_data,test_labels)\n";
	python_script<<"\treturn [train_score,test_score]\n";
	python_script.close();

	PyObject *pName, *pModule, *pValue, *pFunc;//, *pValue;
	char buff[256];
	sprintf(buff,"%d_svm_sklearn",gettid());
    	pName = PyUnicode_DecodeFSDefault(buff);
    
    	pModule = PyImport_Import(pName);
    	if (pModule==NULL) {
		std::cerr<<"Unable to load module "<<pName<<"\n";
        	PyErr_Print();
	    	exit(1);
    	}
    	Py_DECREF(pName);
    	pFunc = PyObject_GetAttrString(pModule,"linearSVM");
    	if (PyCallable_Check(pFunc)) {
        	pValue=PyObject_CallObject(pFunc, NULL);
		_total_sample_train = _train_labels.size();
		_total_sample = _test_labels.size();

		_correct_sample_train=(int)(PyFloat_AsDouble(PyList_GetItem(pValue,0))*_total_sample_train);
		_correct_sample=(int)(PyFloat_AsDouble(PyList_GetItem(pValue,1))*_total_sample);
		Py_DECREF(pValue);
    	} else {
        	PyErr_Print();
    	}
    	
	// Clean up
    	Py_DECREF(pFunc);
    	Py_DECREF(pModule);
    	Py_DECREF(pName);

//	sprintf(buff,"rm -rf %s*",_path);
//	system(buff);
}


void Svm::after_train() {
	
	experiment().print() << "Train samples processed " << _sample_count << std::endl;
	
}

void Svm::before_test() {
	_test_labels.clear();
	_test_samples.clear();
	//_tensor_size=0;
	_correct_sample = 0;
	_total_sample = 0;

}

void Svm::process_test(const std::string& label, const Tensor<float>& sample) {
	process_as_npy_sample(label, sample, _test_labels, _test_samples);
	_sample_count++;
}

void Svm::after_test() {
	//count correct
	call_python_binding(_train_labels, _train_samples, _test_labels, _test_samples);

	experiment().log() << "===SVM===" << std::endl;
	experiment().log() << "train classification rate: " <<
						 (static_cast<float>(_correct_sample_train)/static_cast<float>(_total_sample_train)*100.0) << "% (" <<
						 _correct_sample_train << "/" << _total_sample_train << ")" << std::endl;
	experiment().log() << "classification rate: " <<
						 (static_cast<float>(_correct_sample)/static_cast<float>(_total_sample)*100.0) << "% (" <<
						 _correct_sample << "/" << _total_sample << ")" << std::endl;
	experiment().log() << std::endl;

	before_train();
	before_test();
}

Svm::~Svm() {

}
