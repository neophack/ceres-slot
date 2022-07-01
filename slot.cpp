#include <iostream>
#include <fstream>
#include <sstream>
#include <eigen3/Eigen/Eigen>
#include <ceres/ceres.h>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>

using std::cout;
using std::endl;
using ceres::AutoDiffCostFunction;

//  拟合曲线 y = exp(m*x + c)  x=1:0.01:0.5, m = 3, c = 0.75
// defining a templated object to evaluate the residual

struct costFunctor
{
    costFunctor(double x0, double y0, double x1, double y1, double x2, double y2, double x3, double y3,  int ind,  int jnd): x0_(x0), y0_(y0), x1_(x1), y1_(y1), x2_(x2), y2_(y2), x3_(x3), y3_(y3), ind_(ind), jnd_(jnd)
    {}

    template <typename T>
    bool operator()( const T* const d0_,const T* const d1_, T* residual) const
    {
        // T d0=*(ksi+ind_);
        // T d1=*(ksi+jnd_);
        T d0=d0_[0];
        T d1=d1_[0];
        T nx0= (ceres::cos(d0) * (x0_-x1_) - ceres::sin(d0) * (y0_-y1_)) + x1_;
        T ny0= (ceres::sin(d0) * (x0_-x1_) + ceres::cos(d0) * (y0_-y1_)) + y1_;
        T nx3= (ceres::cos(d1) * (x3_-x2_) - ceres::sin(d1) * (y3_-y2_)) + x2_;
        T ny3= (ceres::sin(d1) * (x3_-x2_) + ceres::cos(d1) * (y3_-y2_)) + y2_;
        residual[0] = ceres::abs((nx0-x1_)*(ny3-y2_)-(nx3-x2_)*(ny0-y1_));
        // T prod = ceres::abs((nx0-x1_)*(ny0-y1_)+(nx3-x2_)*(ny3-y2_));
        // if(prod<10000.){
        //     residual[0] +=prod;
        // }
        //+0.001*(ceres::abs(d0)+ceres::abs(d0));

        return true;
    }

    const double x0_,x1_,x2_,x3_;
    const double y0_,y1_,y2_,y3_;
    const int ind_,jnd_;
};

int main()
{
    struct NeoPoint{
        double x;
        double y;
    };

    struct NeoSlot{
        int p0;
        int p1;
        int p2;
        int p3;
    };
    std::vector<NeoPoint> pl;
    pl.push_back( {10,20});
    pl.push_back( {10,110});
    pl.push_back( {100,130});
    pl.push_back( {110,40});
    pl.push_back( {210,120});
    pl.push_back( {260,50});
    std::vector<NeoSlot> sl;
    sl.push_back( {0,1,2,3});
    sl.push_back( {3,2,4,5});
    int plsize = pl.size();
    int vsize= plsize;
    

    double ksi[vsize];
    for (int i = 0; i < vsize; ++i) {
        ksi[i] = 0;
    }


    ceres::Problem problem;
    for (size_t i = 0; i < sl.size(); ++i)
    {
        NeoSlot st=sl[i];
        NeoPoint p0=pl[st.p0];
        NeoPoint p1=pl[st.p1];
        NeoPoint p2=pl[st.p2];
        NeoPoint p3=pl[st.p3];
        // 三个1, 1, 1分别表示residual的维度， 第一个优化变量c的维度， 第二个优化变量m的维度
        ceres::CostFunction* pCostFunction = new AutoDiffCostFunction<costFunctor, 1, 1,1>(
                new costFunctor(p0.x,p0.y,p1.x,p1.y,p2.x,p2.y,p3.x,p3.y,st.p0,st.p3));
        
        problem.AddResidualBlock(pCostFunction, nullptr, ksi+st.p0,ksi+st.p3);
    }

    // Step2: configure options and solve the optimization problem
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.BriefReport() << endl;

    for (int i = 0; i < vsize; ++i) {

        cout << "ksi: " << i << " -> " << ksi[i] << endl;
    }

    cv::Mat image = cv::Mat(255,255,CV_8UC3,cv::Scalar(255, 255, 255));
    for (size_t i = 0; i < sl.size(); ++i){
        NeoSlot st=sl[i];
        NeoPoint p0=pl[st.p0];
        NeoPoint p1=pl[st.p1];
        NeoPoint p2=pl[st.p2];
        NeoPoint p3=pl[st.p3];

        double d0=ksi[st.p0];
        double d1=ksi[st.p3];
        double nx0= (std::cos(d0) * (p0.x-p1.x) - std::sin(d0) * (p0.y-p1.y)) + p1.x;
        double ny0= (std::sin(d0) * (p0.x-p1.x) + std::cos(d0) * (p0.y-p1.y)) + p1.y;
        double nx3= (std::cos(d1) * (p3.x-p2.x) - std::sin(d1) * (p3.y-p2.y)) + p2.x;
        double ny3= (std::sin(d1) * (p3.x-p2.x) + std::cos(d1) * (p3.y-p2.y)) + p2.y;
        cv::line(image, cv::Point(p0.x,p0.y), cv::Point(p1.x,p1.y),
                        cv::Scalar(255, 0, 128), 1);
        cv::line(image, cv::Point(p1.x,p1.y), cv::Point(p2.x,p2.y),
                        cv::Scalar(0, 0, 255), 1);
        cv::line(image, cv::Point(p2.x,p2.y), cv::Point(p3.x,p3.y),
                        cv::Scalar(255, 0, 128), 1);
        cv::line(image, cv::Point(nx0,ny0), cv::Point(p1.x,p1.y),
                        cv::Scalar(128, 128, 0), 1);
        cv::line(image, cv::Point(p2.x,p2.y), cv::Point(nx3,ny3),
                        cv::Scalar(128, 128, 0), 1);
    }
    cv::imshow("res",image);
    cv::waitKey();

}

