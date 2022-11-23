#ifndef _STDP_BIOLOGICAL_H
#define _STDP_BIOLOGICAL_H

#include "Stdp.h"

namespace stdp {

	class Biological : public STDP {

	public:
		Biological();
		Biological(float alpha, Time tau);

		virtual float process(float w, const Time pre, Time post);
		virtual void adapt_parameters(float factor);
	private:
		float _alpha;
		float _tau;
	};

}
#endif
