#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "kcftracker.hpp"

//#include <dirent.h>

using namespace std;
using namespace cv;

//#define frameToStart 5

int main(){

	bool HOG = true;
	bool FIXEDWINDOW = false;
	bool MULTISCALE = true;
	bool SILENT = true;
	bool LAB = false;

	
	
	// Create KCFTracker object
	KCFTracker tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);
	VideoCapture capture("D:/Car Video/session0_center.avi");
	// Frame readed
	Mat frame;
	Mat resize_frame(400, 600, CV_8UC3);
	//capture.set(CAP_PROP_POS_FRAMES, frameToStart);
	// Tracker results
	Rect result;
	// Frame counter
	int nFrames = 0;
	while (1)
	{
		capture >> frame;
		resize(frame, resize_frame, resize_frame.size());
		// First frame, give the groundtruth to the tracker
		if (nFrames == 0) {
			//imwrite("a.jpg", resize_frame);
			tracker.init( Rect(142,120, 33, 32), resize_frame);
			rectangle(resize_frame, Point(142,120), Point(175,152), Scalar( 0, 0, 255 ), 1, 8 );
			//resultsFile << xMin << "," << yMin << "," << width << "," << height << endl;
		}
		// Update
		else{
			result = tracker.update(resize_frame);
			rectangle(resize_frame, Point( result.x, result.y ), Point( result.x+result.width, result.y+result.height), Scalar( 0, 0, 255 ), 1, 8 );
			//resultsFile << result.x << "," << result.y << "," << result.width << "," << result.height << endl;
		}

		nFrames++;

		if (SILENT){
			imshow("Image", resize_frame);
			waitKey(1);
		}
	}
	//resultsFile.close();

	//listFile.close();

}
