#include <opencv2/contrib/contrib.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <string.h>
#include <strings.h>
#include <iostream>
#include <fstream>


using namespace std;
using namespace cv;

string itos(int i) // convert int to string
{
    stringstream s;
    s << i;
    return s.str();
}

static void dbread(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
    std::ifstream file(filename.c_str(), ifstream::in);

    if (!file) {
        string error = "no valid input file";
        CV_Error(CV_StsBadArg, error);
    }

    string line, path, label;

    while (getline(file, line))
    {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, label);
        if (!path.empty() && !label.empty()){
            images.push_back(imread(path, 0));
            labels.push_back(atoi(label.c_str()));
        }
    }
}




int  FaceTrainer() {
    VideoCapture cap(0);

    if (!cap.isOpened()) {
        cout << "exit" << endl;
        return -1;
    }

    string classifier = "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml";
    CascadeClassifier face_cascade;

    if (!face_cascade.load(classifier)){
            cout << "Error loading haarcascades file" << endl;
            return -1;
    }

    long int count = 0;

    while (true)
    {
        vector<Rect> faces;
        Mat frame;
        Mat graySacleFrame;
        Mat original;

        cap >> frame;
        //count frames;

        int img_width = 92;
        int img_height = 112;

        if (!frame.empty()){
            //clone from original frame
            original = frame.clone();

            //convert image to gray scale and equalize
            cvtColor(original, graySacleFrame, CV_BGR2GRAY);

            //resizing the cropped image to suit to database image sizes
            //detect face in gray image
            face_cascade.detectMultiScale(graySacleFrame, faces, 1.1, 3, 0, cv::Size(92, 112));

            for (int i = 0; i < faces.size(); i++){
                //region of interest
                Rect face_i = faces[i];

                //crop the roi from grya image
                Mat face = graySacleFrame(face_i);


                char c = waitKey(60);

                if (c == 32) {
                    count = count + 1;
                    Mat face_resized;
                    cv::resize(face, face_resized, Size(img_width, img_height), 1.0, 1.0, INTER_CUBIC);
                    string path("test/test"+itos(count)+".jpg");
                    cout << count << "\n";
                    imwrite(path, face_resized);
                }
                //drawing green rectagle in recognize face
                rectangle(original, face_i, CV_RGB(0, 255, 0), 1);
            }

        }

        imshow("Testing window", original);
        char c = waitKey(10);
        if (c == 27) break;
    }

    return 0;
}


int main()
{
    FaceTrainer();
    return 0;
}

