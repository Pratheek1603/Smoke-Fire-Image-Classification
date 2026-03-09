#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>

using namespace cv;
using namespace cv::ml;
using namespace std;
namespace fs = std::filesystem;

Rect yoloToRect(float xc,float yc,float w,float h,int W,int H)
{
    int x = (xc - w/2) * W;
    int y = (yc - h/2) * H;
    int width = w * W;
    int height = h * H;

    return Rect(x,y,width,height);
}

Mat preprocess(Mat roi)
{
    resize(roi,roi,Size(32,32));
    roi.convertTo(roi,CV_32F,1.0/255);

    return roi.reshape(1,1);
}

void loadDataset(string imgDir,string lblDir,
                 Mat &features, Mat &labels)
{
    int counter=0;

    for(auto &file : fs::directory_iterator(imgDir))
    {
        string imgPath=file.path().string();
        string name=file.path().stem().string();

        string labelPath=lblDir+name+".txt";

        Mat img=imread(imgPath);

        if(img.empty()) continue;

        ifstream f(labelPath);
        if(!f.is_open()) continue;

        while(!f.eof())
        {
            int cls;
            float xc,yc,w,h;

            f>>cls>>xc>>yc>>w>>h;

            Rect r=yoloToRect(xc,yc,w,h,img.cols,img.rows);
            r=r&Rect(0,0,img.cols,img.rows);

            if(r.width<20 || r.height<20)
                continue;

            Mat roi=img(r);

            Mat feat=preprocess(roi);

            features.push_back(feat);

            Mat oneHot = Mat::zeros(1,2,CV_32F);

            if(cls==0)
                oneHot.at<float>(0,0)=1.0;
            else
                oneHot.at<float>(0,1)=1.0;

            labels.push_back(oneHot);
        }

        counter++;

        if(counter%500==0)
            cout<<"Loaded "<<counter<<" images"<<endl;
    }
}

float evaluate(Ptr<ANN_MLP> model,
               Mat features,
               Mat labels)
{
    int correct=0;

    for(int i=0;i<features.rows;i++)
    {
        Mat sample=features.row(i);

        Mat pred;
        model->predict(sample,pred);

        Point classIdPred;
        double confPred;

        minMaxLoc(pred,0,&confPred,0,&classIdPred);

        Mat trueRow = labels.row(i);

        Point classIdTrue;
        double confTrue;

        minMaxLoc(trueRow,0,&confTrue,0,&classIdTrue);

        if(classIdPred.x==classIdTrue.x)
            correct++;
    }

    return (float)correct/features.rows;
}

int main()
{
    Mat trainF,trainL;
    Mat valF,valL;
    Mat testF,testL;

    cout<<"Loading training set"<<endl;

    loadDataset("data/train/images/",
                "data/train/labels/",
                trainF,trainL);

    cout<<"Loading validation set"<<endl;

    loadDataset("data/val/images/",
                "data/val/labels/",
                valF,valL);

    cout<<"Loading test set"<<endl;

    loadDataset("data/test/images/",
                "data/test/labels/",
                testF,testL);

    cout<<"Train samples: "<<trainF.rows<<endl;

    Ptr<ANN_MLP> model=ANN_MLP::create();

    Mat layerSizes=(Mat_<int>(1,5) <<
        3072,
        512,
        128,
        32,
        2);

    model->setLayerSizes(layerSizes);

    model->setActivationFunction(
        ANN_MLP::SIGMOID_SYM);

    model->setTrainMethod(
        ANN_MLP::BACKPROP);

    model->setTermCriteria(
        TermCriteria(
            TermCriteria::MAX_ITER+TermCriteria::EPS,
            100,
            0.01));

    cout<<"Training neural network..."<<endl;

    model->train(trainF,
                 ROW_SAMPLE,
                 trainL);

    cout<<"Training finished"<<endl;

    float trainAcc=evaluate(model,trainF,trainL);
    float valAcc=evaluate(model,valF,valL);
    float testAcc=evaluate(model,testF,testL);

    cout<<"Train Accuracy: "<<trainAcc*100<<"%"<<endl;
    cout<<"Validation Accuracy: "<<valAcc*100<<"%"<<endl;
    cout<<"Test Accuracy: "<<testAcc*100<<"%"<<endl;

    model->save("smoke_fire_nn.xml");

    return 0;
}