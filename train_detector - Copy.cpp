#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <filesystem>
#include <fstream>

using namespace cv;
using namespace cv::ml;
using namespace std;
namespace fs = std::filesystem;

// ---------- Convert YOLO box to OpenCV rectangle ----------
Rect yoloToRect(float xc,float yc,float w,float h,int imgW,int imgH)
{
    int x = (xc - w/2) * imgW;
    int y = (yc - h/2) * imgH;
    int width = w * imgW;
    int height = h * imgH;

    return Rect(x,y,width,height);
}

// ---------- Feature Extraction ----------
vector<float> extractFeatures(Mat roi)
{
    resize(roi, roi, Size(64,64));

    vector<float> features;

    // ---- HOG ----
    HOGDescriptor hog(Size(64,64), Size(16,16),
                      Size(8,8), Size(8,8), 9);

    vector<float> hogFeat;
    hog.compute(roi, hogFeat);

    features.insert(features.end(),
                    hogFeat.begin(),
                    hogFeat.end());

    // ---- Color histogram ----
    vector<Mat> channels;
    split(roi, channels);

    int bins = 16;
    float range[] = {0,256};
    const float* histRange = {range};

    for(int i=0;i<3;i++)
    {
        Mat hist;

        calcHist(&channels[i],1,0,
                 Mat(),hist,1,&bins,&histRange);

        hist /= roi.total();

        for(int j=0;j<bins;j++)
            features.push_back(hist.at<float>(j));
    }

    // ---- Texture features ----
    Mat gray;
    cvtColor(roi, gray, COLOR_BGR2GRAY);

    Mat edges;
    Canny(gray, edges, 50,150);

    float edgeDensity =
        countNonZero(edges) /
        (float)edges.total();

    features.push_back(edgeDensity);

    Scalar meanVal, stdVal;
    meanStdDev(gray, meanVal, stdVal);

    float variance = stdVal[0]*stdVal[0];
    features.push_back(variance);

    return features;
}

// ---------- Read YOLO labels ----------
vector<pair<int,Rect>> readLabels(string labelPath,
                                  int imgW,int imgH)
{
    vector<pair<int,Rect>> boxes;

    ifstream file(labelPath);

    if(!file.is_open())
        return boxes;

    while(!file.eof())
    {
        int cls;
        float xc,yc,w,h;

        file>>cls>>xc>>yc>>w>>h;

        Rect r = yoloToRect(xc,yc,w,h,imgW,imgH);

        boxes.push_back({cls,r});
    }

    return boxes;
}

// ---------- Main ----------
int main()
{
    string imgDir = "data/train/images/";
    string lblDir = "data/train/labels/";

    Mat features, labels;

    int count = 0;
    int maxSamples = 3000;

    for(auto &file : fs::directory_iterator(imgDir))
    {
        string imgPath = file.path().string();
        string name = file.path().stem().string();

        string labelPath = lblDir + name + ".txt";

        Mat img = imread(imgPath);

        if(img.empty())
            continue;

        auto boxes =
        readLabels(labelPath,img.cols,img.rows);

        for(auto &b : boxes)
        {
            Rect r = b.second &
                     Rect(0,0,img.cols,img.rows);

            if(r.width<20 || r.height<20)
                continue;

            Mat roi = img(r);

            vector<float> feat =
                extractFeatures(roi);

            Mat row =
            Mat(feat).reshape(1,1);

            features.push_back(row);
            labels.push_back(b.first);

            count++;

            if(count > maxSamples)
                break;
        }

        if(count > maxSamples)
            break;
    }

    features.convertTo(features,CV_32F);

    Ptr<SVM> svm = SVM::create();

    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::RBF);

    svm->setTermCriteria(
        TermCriteria(TermCriteria::MAX_ITER,1000,1e-6));

    svm->train(features, ROW_SAMPLE, labels);

    cout<<"Training finished"<<endl;

    svm->save("fire_smoke_model.xml");
}