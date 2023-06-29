#include "process/SkeletonDetection.h"

using namespace process;

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
#include <opencv2/dnn.hpp>

//
//	SkeletonDetection
//
static RegisterClassParameter<SkeletonDetection, ProcessFactory> _register_1("SkeletonDetection");

SkeletonDetection::SkeletonDetection() : UniquePassProcess(_register_1), _expName(""), _thresh(0), _point_radius(0), _edge_width(0), _draw(0), _onFrame(0), _csv(0), _width(0), _height(0), _depth(0), _conv_depth(0)
{
	add_parameter("thresh", _thresh);
	add_parameter("point_radius", _point_radius);
	add_parameter("edge_width", _edge_width);
	add_parameter("draw", _draw);
	add_parameter("onFrame", _onFrame);
	add_parameter("csv", _csv);
}

SkeletonDetection::SkeletonDetection(std::string expName, float thresh, float point_radius, float edge_width, size_t draw, size_t onFrame, size_t csv) : SkeletonDetection()
{
	parameter<float>("thresh").set(thresh);
	parameter<float>("point_radius").set(point_radius);
	parameter<float>("edge_width").set(edge_width);
	parameter<size_t>("draw").set(draw);
	parameter<size_t>("onFrame").set(onFrame);
	parameter<size_t>("csv").set(csv);
	_expName = expName;
	if (csv == 1)
		std::filesystem::create_directories("Input_frames/" + _expName + "/SD/csv");
	if (draw == 1)
		std::filesystem::create_directories("Input_frames/" + _expName + "/SD/draws");
	_file_path = std::filesystem::current_path();
}

void SkeletonDetection::process_train(const std::string &label, Tensor<float> &sample)
{
	_process(label, sample);
}

void SkeletonDetection::process_test(const std::string &label, Tensor<float> &sample)
{
	_process(label, sample);
}

Shape SkeletonDetection::compute_shape(const Shape &shape)
{
	_height = shape.dim(0);
	_width = shape.dim(1);
	_depth = shape.dim(2);
	_conv_depth = shape.dim(3) > 3 ? shape.dim(3) : 1;
	return Shape({_height, _width, _depth, _conv_depth});
}

void SkeletonDetection::_process(const std::string &label, Tensor<InputType> &in) const
{
	std::vector<cv::Mat> _frames;
	std::vector<cv::Mat> _out_frames;

	Tensor<InputType> out(Shape({_height, _width, _depth, _conv_depth}));

	// This function returns a list of frames that have gone through background subtraction.
	Tensor<float>::tensor_to_matrices(_frames, in);

	cv::Size _inp_size(368, 368);
	cv::Size _frame_size(_width, _height);

	// Specify the paths for the 2 files
#ifdef COCO
	std::string protoFile = "../csnn-simulator/src/process/skeletonPose/coco/pose_deploy_linevec.prototxt";
	std::string weightsFile = "../csnn-simulator/src/process/skeletonPose/coco/pose_iter_440000.caffemodel";
    int pose_pairs [][2] = {{1,0},{1,2},{1,5},{2,3},{3,4},{5,6},{6,7},{1,8},{8,9},{9,10},{1,11},{11,12},{12,13},{0,14},{0,15},{14,16},{15,17}};
	int nPairs = 17;
	int nPoints = 18;
#else
	std::string protoFile = "../csnn-simulator/src/process/skeletonPose/mpii/pose_deploy_linevec.prototxt";
	std::string weightsFile = "../csnn-simulator/src/process/skeletonPose/mpii/pose_iter_160000.caffemodel";
	int pose_pairs [][2] = {{0,1},{1,2},{2,3},{3,4},{1,5},{5,6},{6,7},{1,14},{14,8},{8,9},{9,10},{14,11},{11,12},{12,13}};
	int nPairs = 14;
	int nPoints = 15;
#endif
	// Read the network into Memory
	cv::dnn::Net net = cv::dnn::readNetFromCaffe(protoFile,weightsFile);
	for (int _i = 0; _i < _frames.size(); _i++)
	{
		// Prepare the frame to be fed to the network
		cv::Mat bgr;
		// if (in.shape().dim(2) >= 3)
		// 	bgr = _frames[_i];
		// else
		cvtColor(_frames[_i], bgr, cv::COLOR_GRAY2BGR);
		cv::Mat blob = cv::dnn::blobFromImage(bgr, 1.0 / 255, _inp_size, cv::Scalar(0), false, false);
		// Set the prepared object as the input blob of the network
		net.setInput(blob);
		cv::Mat net_out = net.forward();
		int H = net_out.size[2];
		int W = net_out.size[3];
		cv::Mat output;
		if (_onFrame)
			output = _frames[_i];
		else
			output = cv::Mat::zeros(_frame_size,CV_32FC1);
		// Find the position of the body parts
		std::vector<cv::Point> points(nPoints);
		std::ofstream file;
		if (_csv)
			file.open(_file_path + "/Input_frames/" + _expName + "/SD/csv/csv_" + label + "_" + std::to_string(_i) + ".csv" );
		for (int _n=0; _n < nPoints; _n++)
		{
			// Probability map of corresponding body's part.
			cv::Mat probMap(H, W, CV_32F, net_out.ptr(0,_n));
			cv::Point2f p(-1,-1);
			cv::Point maxLoc;
			double prob;
			minMaxLoc(probMap, 0, &prob, 0, &maxLoc);
			if (prob > _thresh)
			{
				p = maxLoc;
				p.x *= (float)_width / W ;
				p.y *= (float)_height / H ;
				if (_point_radius > 0)
					circle(output, cv::Point((int)p.x, (int)p.y), _point_radius, cv::Scalar(255), -1);
				if (_csv == 1)
					file << std::to_string(_n) + "," + std::to_string(p.x) + "," + std::to_string(p.y) + "\n";
			}
			points[_n] = p;
		}
		if (_csv == 1)
			file.close();

		// Draw the skeleton
		for (int _n = 0; _n < nPairs; _n++)
		{
			// Lookup 2 connected body/hand parts
			cv::Point2f partA = points[pose_pairs[_n][0]];
			cv::Point2f partB = points[pose_pairs[_n][1]];
			if (partA.x<=0 || partA.y<=0 || partB.x<=0 || partB.y<=0)
				continue;
			if (_edge_width > 0)
				line(output, partA, partB, cv::Scalar(255), _edge_width);
		}
		_out_frames.push_back(output);
	}
	
	Tensor<float>::matrices_to_tensor(_out_frames, out);
	if (_draw == 1)
		Tensor<float>::draw_tensor(_file_path + "/Input_frames/" + _expName + "/SD/draws/draws_" + label + "_", out);
	in = out;
}