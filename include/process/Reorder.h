#ifndef _PROCESS_REORDER_H
#define _PROCESS_REORDER_H

#include "Process.h"
#include "tool/Operations.h"

namespace process
{
	class UniformReorder : public UniquePassProcess
	{

	public:
		UniformReorder();
		UniformReorder(float min_time, float max_time);

		virtual Shape compute_shape(const Shape &shape);
		virtual void process_train(const std::string &label, Tensor<float> &sample);
		virtual void process_test(const std::string &label, Tensor<float> &sample);

	private:
		void _process(Tensor<float> &sample) const;

		float _min_time;
		float _max_time;

	};

}

#endif
