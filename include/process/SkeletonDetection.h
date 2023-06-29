#ifndef _PROCESS_SKELETON_DETECTION_H
#define _PROCESS_SKELETON_DETECTION_H

#include <filesystem>
#include <iostream>
#include "Process.h"
#include "NumpyReader.h"

namespace process
{

	namespace _priv
	{
		class Skeleton
		{

		public:
			Skeleton() = delete;
		};
	}

	/**
	 * @brief Extracts the skeleton of human body using COCO or MPII model.
	 * 
	 * @param expName The name of the expirement (set this to experiment.name()), in order to save the drawn value in a folder that has this name.
	 * @param threshold A value used to tell if a point should be taken in account or not.
	 * @param point_radius The radius of points. Set this to 0 if you do not want to draw points.
	 * @param edge_width The width of edges. Set this to 0 if you do not want to draw edges.
	 * @param draw A flag that draws the skeleton in the build folder in a folder called Input_frames.
	 * @param onFrame A flag that tells if the skeleton should be drawn on the frame or on an empty image.
	 * @param csv A flag that adds the coordinates of points in the build folder in a folder called Input_frames.
	 */
	class SkeletonDetection : public UniquePassProcess
	{

	public:
		SkeletonDetection();
		SkeletonDetection(std::string expName, float thresh = 0.1, float point_radius = 2, float edge_width = 1, size_t draw = 0, size_t onFrame = 0, size_t csv = 0);

		virtual Shape compute_shape(const Shape &shape);
		virtual void process_train(const std::string &label, Tensor<float> &sample);
		virtual void process_test(const std::string &label, Tensor<float> &sample);

	private:
		void _process(const std::string & label, Tensor<float> &in) const;
		float _point_radius;
		float _edge_width;
		size_t _draw;
		size_t _onFrame;
		size_t _csv;
		std::string _expName;
		// threshold to decide if a point is accurate enough to be taken in account
		float _thresh;
		// To save the drawings
		std::string _file_path;

		size_t _width;
		size_t _height;
		size_t _depth;
		size_t _conv_depth;
	};

}
#endif
