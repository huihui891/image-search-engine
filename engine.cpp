#include <iostream>
#include <vector>
#include <Windows.h>
#include <fstream>
#include <string>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;
using namespace std;

#define GLCM_DIS 3  //灰度共生矩阵的统计距离
#define GLCM_CLASS 16 //计算灰度共生矩阵的图像灰度值等级化
#define GLCM_ANGLE_HORIZATION 0  //水平
#define GLCM_ANGLE_VERTICAL   1	 //垂直
#define GLCM_ANGLE_DIGONAL    2  //对角
#define NUM_PIC 10   //图片的数目
#define THRESHOLD 0.5;
fstream fout("data",ios_base::out|ios_base::app);

string path("E:\\project\\test\\test\\images\\");	

int calGLCM(IplImage* bWavelet,int angleDirection,vector<double>& featureVector)
{
	int i,j;
	int width,height;

	if(NULL == bWavelet)
		return 1;

	width = bWavelet->width;
	height = bWavelet->height;

	int * glcm = new int[GLCM_CLASS * GLCM_CLASS];
	int * histImage = new int[width * height];

	if(NULL == glcm || NULL == histImage)
		return 2;

	//灰度等级化---分GLCM_CLASS个等级
	uchar *data =(uchar*) bWavelet->imageData;
	for(i = 0;i < height;i++){
		for(j = 0;j < width;j++){
			histImage[i * width + j] = (int)(data[bWavelet->widthStep * i + j] * GLCM_CLASS / 256);
		}
	}

	//初始化共生矩阵
	for (i = 0;i < GLCM_CLASS;i++)
		for (j = 0;j < GLCM_CLASS;j++)
			glcm[i * GLCM_CLASS + j] = 0;

	//计算灰度共生矩阵
	int w,k,l;
	//水平方向
	if(angleDirection == GLCM_ANGLE_HORIZATION)
	{
		for (i = 0;i < height;i++)
		{
			for (j = 0;j < width;j++)
			{
				l = histImage[i * width + j];
				if(j + GLCM_DIS >= 0 && j + GLCM_DIS < width)
				{
					k = histImage[i * width + j + GLCM_DIS];
					glcm[l * GLCM_CLASS + k]++;
				}
				if(j - GLCM_DIS >= 0 && j - GLCM_DIS < width)
				{
					k = histImage[i * width + j - GLCM_DIS];
					glcm[l * GLCM_CLASS + k]++;
				}
			}
		}
	}
	//垂直方向
	else if(angleDirection == GLCM_ANGLE_VERTICAL)
	{
		for (i = 0;i < height;i++)
		{
			for (j = 0;j < width;j++)
			{
				l = histImage[i * width + j];
				if(i + GLCM_DIS >= 0 && i + GLCM_DIS < height) 
				{
					k = histImage[(i + GLCM_DIS) * width + j];
					glcm[l * GLCM_CLASS + k]++;
				}
				if(i - GLCM_DIS >= 0 && i - GLCM_DIS < height) 
				{
					k = histImage[(i - GLCM_DIS) * width + j];
					glcm[l * GLCM_CLASS + k]++;
				}
			}
		}
	}
	//对角方向
	else if(angleDirection == GLCM_ANGLE_DIGONAL)
	{
		for (i = 0;i < height;i++)
		{
			for (j = 0;j < width;j++)
			{
				l = histImage[i * width + j];

				if(j + GLCM_DIS >= 0 && j + GLCM_DIS < width && i + GLCM_DIS >= 0 && i + GLCM_DIS < height)
				{
					k = histImage[(i + GLCM_DIS) * width + j + GLCM_DIS];
					glcm[l * GLCM_CLASS + k]++;
				}
				if(j - GLCM_DIS >= 0 && j - GLCM_DIS < width && i - GLCM_DIS >= 0 && i - GLCM_DIS < height)
				{
					k = histImage[(i - GLCM_DIS) * width + j - GLCM_DIS];
					glcm[l * GLCM_CLASS + k]++;
				}
			}
		}
	}

	//计算特征值
	double entropy = 0,energy = 0,contrast = 0,homogenity = 0;
	for (i = 0;i < GLCM_CLASS;i++)
	{
		for (j = 0;j < GLCM_CLASS;j++)
		{
			//熵
			if(glcm[i * GLCM_CLASS + j] > 0)
				entropy -= glcm[i * GLCM_CLASS + j] * log10(double(glcm[i * GLCM_CLASS + j]));
			//能量
			energy += glcm[i * GLCM_CLASS + j] * glcm[i * GLCM_CLASS + j];
			//对比度
			contrast += (i - j) * (i - j) * glcm[i * GLCM_CLASS + j];
			//一致性
			homogenity += 1.0 / (1 + (i - j) * (i - j)) * glcm[i * GLCM_CLASS + j];
		}
	}
	//返回特征值
	featureVector.push_back(entropy);
	featureVector.push_back(energy);
	featureVector.push_back(contrast);
	featureVector.push_back(homogenity);

	delete[] glcm;
	delete[] histImage;
	return 0;
}
int calDistance(const vector<double>& f1,const vector<double>& f2)
{
	assert(f1.size() == f2.size());
	double sum = 0.0;
	//欧氏距离
// 	for (int i = 0; i < f1.size(); i++)
// 		sum += (f1[i] - f2[i])*(f1[i] - f2[i]);
// 	return sqrtl(sum);

	//曼哈顿距离
// 	for (int i = 0; i < f1.size(); i++)
// 		sum += abs(f1[i] - f2[i]);
// 	return sum;

	//切比雪夫距离
// 	vector<double> absDiff;
// 	absDiff.reserve(f1.size());
// 	for (int i = 0; i < f1.size(); i++)
// 		absDiff.push_back(abs(f1[i] - f2[i]));
// 	sort(absDiff.begin(),absDiff.end());//low-high
// 	return absDiff[f1.size()-1];

	//加权欧氏距离
// 	vector<double> stdd;
// 	stdd.reserve(f1.size());
// 	for (int i = 0; i < f2.size(); i++)
// 	{
// 		int mean = (f1[i]+f2[i])/2;
// 		stdd.push_back(((p1[i]-mean)*(p1[i]-mean)+(p2[i]-mean)*(p1[2]-mean))/2);
// 	}
// 	for (int i = 0; i < f2.size(); i++)
// 		sum += (p1[i] - p2[i])*(p1[i] - p2[i])/stdd[i];
// 
// 	return sqrtl(sum);



}

int main()
{
	//训练
	string ImgName;//图片名(绝对路径)
	ifstream fin("E:\\project\\test\\test\\images\\pic.txt");//正样本图片的文件名列表

	for(int num=0; num<NUM_PIC && getline(fin,ImgName); num++)
	{
		cout<<"处理："<<ImgName<<endl;
		ImgName = path + ImgName;//加上正样本的路径名
		Mat img = imread(ImgName);//读取图片
		if (!img.data)
			cout << "load error!";

		//cout << img.channels() << endl;
		Mat gray;
		if (img.channels() == 3)
		{		
			cvtColor(img,gray,CV_BGR2GRAY);
		}


		IplImage ipl = IplImage(gray);
		vector<double> feature;
		feature.reserve(4);
		calGLCM(&ipl,2,feature);
		fout << feature[0] << " " << feature[1] << " " << feature[2] << " " << feature[3] << endl;
	}

	//检测
	double mean,std;
	vector<double> model;
	model.reserve(4);

	//导入图片 计算特征向量
	Mat test = imread("test.jpg");
	if (!test.data)
		cout << "load error!";

	Mat gray;
	if (test.channels() == 3)
	{
		cvtColor(test,gray,CV_BGR2GRAY);
	}

	IplImage ipl = IplImage(gray);
	vector<double> feature;
	feature.reserve(4);
	calGLCM(&ipl,2,feature);
	

	//计算向量之间的距离
//	if (calDistance() > THRESHOLD)
		


	
	return 1;
}
