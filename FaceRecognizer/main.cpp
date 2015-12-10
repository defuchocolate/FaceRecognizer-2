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
#include <iomanip>

using namespace std;
using namespace cv;

string itos(int i) // convert int to string
{
    stringstream s;
    s << i;
    return s.str();
}

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }
    string line, path, classlabel;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
            images.push_back(imread(path, 0));
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}

static void read_listNames(const string& filename, vector<string>& names, vector<int>& labels,  char separator = ';') {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }
    string line, name, classlabel;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, name, separator);
        getline(liness, classlabel);
        if(!name.empty() && !classlabel.empty()) {
            names.push_back(name);
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}


void fisherFaceTrainer(){
    /*in this two vector we put the images and labes for training*/
    vector<Mat> images;
    vector<int> labels;

    try{
        string filename = "../Faces/FaceList.csv";
        read_csv(filename, images, labels);

        cout << "size of the images is " << images.size() << endl;
        cout << "size of the labes is " << labels.size() << endl;
        cout << "Training begins...." << endl;
    }
    catch (cv::Exception& e){
        cerr << " Error opening the file " << e.msg << endl;
        exit(1);
    }


    Ptr<FaceRecognizer> model = createFisherFaceRecognizer();

    model->train(images, labels);

    int height = images[0].rows;

    model->save("../Faces/FisherFaces.yml");

    cout << "Training finished...." << endl;

    waitKey(10000);
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


                    system("mkdir test");
                    string path("test/"+itos(count)+".pgm");
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

int  FaceRecognition(){


    vector<string> names;
    vector<int> labels;
    read_listNames("../Faces/PersonNames.txt", names, labels);

    cout << "start recognizing..." << endl;

    //load pre-trained data sets
    Ptr<FaceRecognizer>  model = createFisherFaceRecognizer();
    model->load("../Faces/FisherFaces.yml");
    int img_width = 92;
    int img_height = 112;

    string classifier = "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml";

    CascadeClassifier face_cascade;

    if (!face_cascade.load(classifier)){
        cout << " Error loading haarcascades file" << endl;
        return -1;
    }

    VideoCapture cap(0);

    if (!cap.isOpened())
    {
        cout << "exit" << endl;
        return -1;
    }

    long count = 0;

    while (true)
    {
        vector<Rect> faces;
        Mat frame;
        Mat graySacleFrame;
        Mat original;

        cap >> frame;

        //count frames
        count++;

        if (!frame.empty()){

            //clone from original frame
            original = frame.clone();

            //convert image to gray scale and equalize
            cvtColor(original, graySacleFrame, CV_BGR2GRAY);

            //detect face in gray image
            face_cascade.detectMultiScale(graySacleFrame, faces, 1.1, 3, 0, cv::Size(90, 90));

            //number of faces detected
            cout << faces.size() << " faces detected" << endl;
            string frameset = std::to_string(count);
            string faceset = std::to_string(faces.size());

            //person name
            string Pname = "";

            for (int i = 0; i < faces.size(); i++)
            {
                //region of interest
                Rect face_i = faces[i];

                //crop the roi from grya image
                Mat face = graySacleFrame(face_i);

                //resizing the cropped image to suit to database image sizes
                Mat face_resized;
                resize(face, face_resized, Size(img_width, img_height), 1.0, 1.0, INTER_CUBIC);

                //recognizing what faces detected
                int label = -1;
                double confidence = 0;
                model->predict(face_resized, label, confidence);

                cout << " confidencde " << confidence << endl;

                //drawing green rectagle in recognize face
                rectangle(original, face_i, CV_RGB(0, 255, 0), 1);

                //string text = "Detected";

                for (int i = 0; i < labels.size(); i++){
                    if (label == labels[i] && confidence < 1100){
                        Pname = names[i];
                        break;
                    }
                    else{
                        Pname = "unknown";
                    }
                }


                int pos_x = std::max(face_i.tl().x - 10, 0);
                int pos_y = std::max(face_i.tl().y - 10, 0);

                //name the person who is in the image
                //putText(original, text, Point(pos_x, pos_y), FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0, 255, 0), 1.0);

            }


            putText(original, "Frames: " + frameset, Point(30, 60), CV_FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0, 255, 0), 1.0);
            putText(original, "Person: " + Pname, Point(30, 90), CV_FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0, 255, 0), 1.0);
            //display to the winodw
            imshow("Capture", original);

            cout << "model infor " << model->getDouble("threshold") << endl;

        }
        if (waitKey(30) >= 0) break;
    }
}




int main()
{
    //FaceTrainer();
    fisherFaceTrainer();
    FaceRecognition();
    return 0;
}
