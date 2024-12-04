#include "execution/TrainingExecution.h"
#include "Math.h"

TrainingExecution::TrainingExecution(ExperimentType& experiment) :
	_experiment(experiment), _train_set() {

	// Create the model sub-directory storing trained parameters for the trainable processes
	mkdir(_experiment.model_path().c_str(), S_IRWXU | S_IRWXG | S_IRWXO);
}

void TrainingExecution::process(size_t refresh_interval) {
	_load_data();

	for(size_t i=0; i<_experiment.process_number(); i++) {
		_experiment.print() << "Process " << _experiment.process_at(i).factory_name() << "." << _experiment.process_at(i).class_name();
		if(!_experiment.process_at(i).name().empty()) {
			_experiment.print() << " (" << _experiment.process_at(i).name() << ")";
		}
		_experiment.print() << std::endl;

		_process_train_data(_experiment.process_at(i), _train_set, refresh_interval);
		_process_output(i);
		// Save trained parameters
		std::string process_save_path = _experiment.model_path() + "/" + _experiment.process_at(i).factory_name() + "." + _experiment.process_at(i).class_name();
		if (!_experiment.process_at(i).name().empty()) {
			process_save_path += "." + _experiment.process_at(i).name();
		}
		process_save_path += "/";
		// Create the process sub-directory
		mkdir(process_save_path.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);
		bool saved = _experiment.process_at(i).save_params(process_save_path);
		if (saved) {
			_experiment.log() << "Save trained parameters at " << process_save_path << std::endl;
		}
		else {
			// Remove process sub-directory if no trained parameters saved (process not trainable)
			rmdir(process_save_path.c_str());
		}
	}

	_train_set.clear();
}

Tensor<Time> TrainingExecution::compute_time_at(size_t i) const {
	throw std::runtime_error("Unimplemented");
}

void TrainingExecution::_load_data() {
	for(Input* input : _experiment.train_data()) {
		size_t count = 0;
		while(input->has_next()) {
			_train_set.push_back(input->next());
			count ++;
		}
		_experiment.log() << "Load " << count << " train samples from " << input->to_string() << std::endl;
		input->close();
	}
}

void TrainingExecution::_process_train_data(AbstractProcess& process, std::vector<std::pair<std::string, Tensor<float>>>& data, size_t refresh_interval) {
	size_t n = process.train_pass_number();

	if(n == 0) {
		throw std::runtime_error("train_pass_number() should be > 0");
	}

	refresh_interval=10;
	for(size_t i=0; i<n; i++) {
		for(size_t j=0; j<data.size(); j++) {
			if ((j+1)%100==0)
				_experiment.log() << "processing sample " <<j<< " out of "<<data.size()<<"\n";
			process.process_train_sample(data[j].first, data[j].second, i, j, data.size());

			if(i == n-1 && data[j].second.shape() != process.shape()) {
				throw std::runtime_error("Unexpected shape (actual: "+data[j].second.shape().to_string()+", expected: "+process.shape().to_string()+")");
			}

			_experiment.tick(process.index(), i*data.size()+j);

			if((i*data.size()+j) % refresh_interval == 0) {
				_experiment.refresh(process.index());
			}

		}
	}
}

void TrainingExecution::_process_output(size_t index) {
	for(size_t i=0; i<_experiment.output_count(); i++) {
		if(_experiment.output_at(i).index() == index) {
			Output& output = _experiment.output_at(i);

			std::vector<std::pair<std::string, Tensor<float>>> output_train_set;

			for(std::pair<std::string, Tensor<float>>& entry : _train_set) {
				output_train_set.emplace_back(entry.first, output.converter().process(entry.second));
			}

			for(Process* process : output.postprocessing()) {
				_experiment.print() << "Process " << process->class_name() << std::endl;
				_process_train_data(*process, output_train_set, std::numeric_limits<size_t>::max());
			}

			for(Analysis* analysis : output.analysis()) {

				_experiment.log() << output.name() << ", analysis " << analysis->class_name() << ":" << std::endl;

				size_t n = analysis->train_pass_number();

				for(size_t j=0; j<n; j++) {
					analysis->before_train_pass(j);
					for(std::pair<std::string, Tensor<float>>& entry : output_train_set) {
						analysis->process_train_sample(entry.first, entry.second, j);
					}
					analysis->after_train_pass(j);
				}

				if(n == 0) {
					analysis->after_test();
				}
			}
		}
	}
}
