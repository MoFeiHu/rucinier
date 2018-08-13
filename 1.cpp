#include <iostream>
#include<opencv2\opencv.hpp>
#include<opencv2\imgproc\imgproc.hpp>
#include "MixtureOfGaussianV2.h"
#include "IBGS.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include "kcftracker.hpp"
#include <math.h>
using  namespace std;
using  namespace cv;

#define resizedHeight 480
#define resizedWidth  600

//#define frameTostart 20

int main()
{
	bool HOG = true;
	bool FIXEDWINDOW = false;
	bool MULTISCALE = true;
	bool SILENT = true;
	bool LAB = false;
	vector<KCFTracker> trackers;		//创建一个KCFTracker类的数组
	
	VideoCapture capture("D:/Car Video/session0_center.avi");
	if (!capture.isOpened())
	{
		cout << "No video input\n" << endl;
		return -1;
	}
	vector<Rect> RESULT;
	Rect result;
	Rect r;
	IBGS *p;
	p = new MixtureOfGaussianV2();
	int pause = 0;
	Mat img;
	Mat img_resized(resizedHeight, resizedWidth, CV_8UC3);
	//capture.set(CAP_PROP_POS_FRAMES, frameTostart);			//设置起始帧；CAP_PROP_POS_FRAMES：单位为帧数的位置（仅对视频文件有效）
	//MixtureOfGaussianV2 mog;
	int nFrames = 0;
	while (!pause)
	{
		nFrames++;
		capture >> img;
		if (img.empty())
			break;

		resize(img, img_resized, img_resized.size());

		Mat img_mask;
		Mat img_bkgmodel;
		p->process(img_resized, img_mask, img_bkgmodel);
		//cout << "a" << img_mask.channels() << endl;
		//imwrite("1.jpg", img_mask);
		Mat dst;
		Mat element = getStructuringElement(MORPH_RECT, Size(7, 7));
		morphologyEx(img_mask, dst, MORPH_CLOSE, element);
		morphologyEx(dst, dst, MORPH_OPEN, element);
		imshow("形态学处理后", dst);

		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;
		vector<vector<Point>> lunkuo;
		findContours(dst, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_NONE);

		int cmin = 50;
		//vector<vector<Point>>::iterator itc;
		//itc = contours.begin();
		for (int i = 0; i < contours.size();i++)
		{
			if (contours[i].size()>cmin)
				lunkuo.push_back(contours[i]);
		}

		if (nFrames == 1)
		{
			for (int i = 0; i < lunkuo.size(); i++)
			{
				r = boundingRect(lunkuo[i]);
				KCFTracker A(HOG, FIXEDWINDOW, MULTISCALE, LAB);
				A.init(Rect(r.x, r.y, r.width, r.height), img_resized);
				trackers.push_back(A);
				//rectangle(img_resized, Point(r.x, r.y), Point(r.x + r.width, r.y + r.height), Scalar(0, 0, 255), 2, 8);
			}
			/*for (int j = 0; j < trackers.size(); j++)
			{
				result = trackers[j].update(img_resized);
				RESULT.push_back(result);
			}*/
		}

		if (nFrames != 1)
		{
			for (int i = 0; i < lunkuo.size(); i++)
			{
				bool flag = false;
				double d,dis;
				r = boundingRect(lunkuo[i]);
				double x1 = r.x + 0.5*r.width;
				double y1 = r.y + 0.5*r.height;
				for (int j = 0; j < trackers.size(); j++)
				{
					double x2 = RESULT[j].x + 0.5*RESULT[j].width;
					double y2 = RESULT[j].y + 0.5*RESULT[j].height;
					double x = x2 - x1;
					double y = y2 - y1;
					double d = x*x + y*y;
					dis = sqrt(d);		
					if (dis < 100)
					{
						flag = true;
						RESULT.pop_back();
						break;
					}
				}
				
				if (flag) continue;
				
				KCFTracker A(HOG, FIXEDWINDOW, MULTISCALE, LAB);
				A.init(Rect(r.x, r.y, r.width, r.height), img_resized);
				trackers.push_back(A);
				rectangle(img_resized, Point(r.x, r.y), Point(r.x + r.width, r.y + r.height), Scalar(0,255, 0), 2, 8);
		    } 
		}
					
		for (int j = 0; j < trackers.size(); j++)
		{
			result = trackers[j].update(img_resized);
			RESULT.push_back(result);
			rectangle(img_resized, Point(result.x, result.y),
				Point(result.x + result.width, result.y + result.height), Scalar(0,255,0), 1, 8);
		}
				
		if (SILENT)
		{
			imshow("目标跟踪", img_resized);
			//waitKey(1);
		}
		if (cvWaitKey(10) == 'q')
			pause = !pause;

	}

		//namedWindow("原图加边框", WINDOW_NORMAL);
		//imshow("原图加边框", img_resized);
		//cout << "该帧轮廓数量为：" << contours.size() << endl;
		delete p;
		cvDestroyAllWindows();
		capture.release();
}
		

