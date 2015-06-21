#include <iostream>
#include <fstream>
#include <string>
#include "opencv2/opencv_modules.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/stitching/stitcher.hpp"

using namespace cv;
using namespace std;

int main()
{
  VideoCapture cam0(1);
  VideoCapture cam1(2);

  vector<Mat> frame(2);
  Mat pano;
  Stitcher stitcher = Stitcher::createDefault(false);
  Stitcher::Status status;

  while(true)
  {
    cam0.read(frame[0]);
    cam1.read(frame[1]);
  
    status = stitcher.stitch(frame, pano);

    if (status != Stitcher::OK)
    {
        cout << "Can't stitch images, error code = " << int(status) << endl;
    }
    else {
      imshow("result", pano);
      imshow("cam1", frame[0]);
      imshow("cam2", frame[1]);
      waitKey();
    }
  }
}


