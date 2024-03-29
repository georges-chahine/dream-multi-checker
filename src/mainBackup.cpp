﻿#include <stdio.h>
#include <math.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <sstream>
#include <fstream>
#include "XMeansT.h"
#include "DMeansT.h"
#include "LinearMath/Quaternion.h"
#include "LinearMath/Matrix3x3.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include "tf_conversions/tf_eigen.h"
using namespace std;
using namespace Clustering;

static const unsigned int N = 2;
struct se3 {
    tf::Vector3 t;
    tf::Quaternion q;
    se3() {}
    se3(const tf::Vector3 & t, const tf::Quaternion & q) : t(t), q(q) {}
};
typedef tf::Quaternion DI;
typedef se3 DISE3;

/*
 * template <class stream>
stream & operator<<(stream & out, const DI & q) {
    // out << q.x() << " " << q.y() << " " << q.z() << " " << q.w();
    tf::Matrix3x3 R(q);
    double roll,pitch,yaw;
    R.getRPY(roll,pitch,yaw);
    out << roll << " " << pitch << " " << yaw;
    return out;
}
*/



template <class stream>
stream & operator<<(stream & out, const DISE3 & t) {
    out << t.t.x() << " " << t.t.y() << " " << t.t.z() << " ";
#if 0
    out << t.q.x() << " " << t.q.y() << " " << t.q.z() << " " << t.q.w();
#else
    tf::Matrix3x3 R(t.q);
    double roll,pitch,yaw;
    R.getRPY(roll,pitch,yaw);
    out << roll << " " << pitch << " " << yaw;
#endif
    return out;
}


typedef std::vector<DI> DataSet;
typedef std::vector<se3> DataSetSe3;
typedef DataSet::const_iterator DataSetIterator;
typedef DataSetSe3::const_iterator DataSetSe3Iterator;





struct DataMetric {
    static double trans_weight;
    static double rot_weight;
    double operator()(const DISE3 & a, const DISE3 & b) const {
        tf::Quaternion q = a.q * b.q.inverse();
        return trans_weight*(a.t-b.t).length() + rot_weight*q.getAngle();
    }
};

//double DataMetric::trans_weight = 1.0/0.15;
//double DataMetric::rot_weight = 1.0/0.1;

double DataMetric::trans_weight = 1.0/0.3;
double DataMetric::rot_weight = 1.0/0.1;
/*

struct DataMetric {
    double operator()(const DI & a, const DI & b) const {
        tf::Quaternion q = a * b.inverse();
        return q.getAngle();
    }
};
*/

tf::Vector3 QLog(const tf::Quaternion & q) {
    return q.getAngle() * q.getAxis();
}

tf::Quaternion QExp(tf::Vector3 v) {
    double angle = v.length();
    if (fabs(angle)<1e-4) {
        return tf::Quaternion::getIdentity();
    } else {
        v /= angle;
        return tf::Quaternion(v,angle);
    }
}

// Random displacement quaternion around 0, using the manifold structure
template <class Generator>
tf::Quaternion dQRand(double sigma, Generator & rge) {
    std::normal_distribution<double> dis(0,1);
    tf::Vector3 l_dq(dis(rge)*sigma,dis(rge)*sigma,dis(rge)*sigma);// == log(dq)
    return QExp(l_dq);
}

// Uniform random quaternion with uniform roll pitch yaw
template <class Generator>
tf::Quaternion QRand(Generator & rge) {
    std::uniform_real_distribution<double> dis(-M_PI,M_PI);
    tf::Quaternion res;
    res.setRPY(dis(rge),dis(rge),dis(rge));
    return res;
}

// Random translation
template <class Generator>
tf::Vector3 dVRand(double sigma, Generator & rge) {
    std::normal_distribution<double> dis(0,1);
    return tf::Vector3(dis(rge)*sigma,dis(rge)*sigma,dis(rge)*sigma);
}

// Uniform random quaternion with uniform roll pitch yaw
template <class Generator>
tf::Vector3 VRand(Generator & rge) {
    std::uniform_real_distribution<double> dis(-1,1);
    return tf::Vector3(dis(rge),dis(rge),dis(rge));
}



struct DataAggregator {
    size_t max_iter;
    double tol;
    DataAggregator(double tol=5e-3,size_t max_iter=100) : max_iter(max_iter), tol(tol) {}

    DISE3 operator()(DataSetSe3Iterator ds_begin, MemberSetIterator mb_begin, MemberSetIterator mb_end) const {
        if (mb_begin == mb_end) {
            return se3(tf::Vector3(0,0,0),tf::Quaternion::getIdentity());
        }
        size_t count = 0;

        // Translation mean
        tf::Vector3 tmean(0,0,0);
        count = 0;
        for (MemberSetIterator mb_it = mb_begin; mb_it != mb_end; mb_it++) {
            tmean += (ds_begin + *mb_it)->t;
            count += 1;
        }
        tmean /= count;

        // Rotation mean
        tf::Quaternion qmean = (ds_begin + *mb_begin)->q;
        // std::cout << "Aggregation " << std::endl;
        for (size_t i=0;i<max_iter;i++) {
            // std::cout << i << std::endl;
            tf::Quaternion qmean_prev(qmean);
            tf::Vector3 e(0,0,0);
            count = 0;
            for (MemberSetIterator mb_it = mb_begin; mb_it != mb_end; mb_it++) {
                const tf::Quaternion & q = (ds_begin + *mb_it)->q;
                e += QLog(q * qmean.inverse());
                count += 1;
            }
            e /= count;
            // std::cout << count << " " << e.x() << " " << e.y() << " " << e.z() << std::endl;
            qmean = QExp(e) * qmean;
            // tf::Quaternion qdiff = qmean * qmean_prev.inverse();
            // std::cout << qmean_prev << " -> " << qmean << " " << qdiff << "=>" << (qmean * qmean_prev.inverse()).getAngle() << std::endl;
            if ((qmean * qmean_prev.inverse()).getAngle() < tol) {
                break;
            }
        }
        // std::cout << "Aggregation done" << std::endl;
        return se3(tmean,qmean);
    }
};

class CenterSplitter {
protected:
    std::normal_distribution<double> dis;
public:
    CenterSplitter() : dis(0,1) {}
    template <class Generator = std::default_random_engine>
    std::pair<DISE3,DISE3> operator()(const DISE3 & c, double sigma, Generator & rge) {
        tf::Vector3 dt(dis(rge)*sigma,dis(rge)*sigma,dis(rge)*sigma);
        tf::Vector3 l_dq(dis(rge)*sigma,dis(rge)*sigma,dis(rge)*sigma);// == log(dq)
        tf::Quaternion dq = QExp(l_dq);

        tf::Quaternion c1 = dq * c.q;
        tf::Quaternion c2 = dq.inverse() * c.q;
        return std::pair<se3,se3>(se3(c.t+dt,c1),se3(c.t-dt,c2));
    }
};

/*
struct DataAggregator {
    size_t max_iter;
    double tol;
    DataAggregator(double tol=5e-4,size_t max_iter=100) : max_iter(max_iter), tol(tol) {}

    DI operator()(DataSetIterator ds_begin, MemberSetIterator mb_begin, MemberSetIterator mb_end) const {
        if (mb_begin == mb_end) {
            return tf::Quaternion::getIdentity();
        }
        tf::Quaternion qmean = *(ds_begin + *mb_begin);
        // std::cout << "Aggregation " << std::endl;
        for (size_t i=0;i<max_iter;i++) {
            // std::cout << i << std::endl;
            tf::Quaternion qmean_prev(qmean);
            tf::Vector3 e(0,0,0);
            size_t count = 0;
            for (MemberSetIterator mb_it = mb_begin; mb_it != mb_end; mb_it++) {
                const DI & q = *(ds_begin + *mb_it);
                e += QLog(q * qmean.inverse());
                count += 1;
            }
            e /= count;
            // std::cout << count << " " << e.x() << " " << e.y() << " " << e.z() << std::endl;
            qmean = QExp(e) * qmean;
            // std::cout << qmean_prev << " -> " << qmean << " " << (qmean_prev.inverse() * qmean).getAngle() << std::endl;
            if ((qmean_prev.inverse() * qmean).getAngle() < tol) {
                break;
            }
        }
        // std::cout << "Aggregation done" << std::endl;
        return qmean;
    }
};

class CenterSplitter {
protected:
    std::normal_distribution<double> dis;
public:
    CenterSplitter() : dis(0,1) {}
    template <class Generator = std::default_random_engine>
    std::pair<DI,DI> operator()(const DI & c, double sigma, Generator & rge) {
        tf::Vector3 l_dq(dis(rge)*sigma,dis(rge)*sigma,dis(rge)*sigma);// == log(dq)
        tf::Quaternion dq = QExp(l_dq);

        DI c1 = dq * c;
        DI c2 = dq.inverse() * c;
        return std::pair<DI,DI>(c1,c2);
    }
};
*/
// > means newer
Eigen::Matrix4d parseData(int srcKF, int dstKF, int srcT, int dstT, vector<vector<string>> transforms)
{
    bool invFlag=false;
    //  std::cout<<"dstKF is " <<dstKF<<std::endl;
    if (dstT>srcT){
        int tempKF, tempT;
        tempKF=srcKF;
        tempT=srcT;
        srcKF=dstKF;
        srcT=dstT;
        dstKF=tempKF;
        dstT=tempT;
        invFlag=true;
        // std::cout<<"inverted"<<std::endl;

    }

    Eigen::Matrix4d transform=Eigen::Matrix4d::Identity();

    for (unsigned int i=0; i<transforms.size();i++){

        if (transforms[i][0]=="%KF"){
            int sKF=stoi(transforms[i+1][0],nullptr,0);
            int sT=stoi(transforms[i+1][1],nullptr,0);
            if ((sKF==dstKF) && (sT==dstT)){
                unsigned int j=2;
                while (true){
                    if (transforms[i+j][0]=="%KF"){break;}

                    int cKF=stoi(transforms[i+j][0],nullptr,0);
                    int cT=stoi(transforms[i+j][1],nullptr,0);
                    // std::cout<<"cT is "<<cT<<" dstT is "<<dstT<<std::endl;


                    if ((cKF==srcKF) && (cT==srcT)){
                        //   std::cout<<"ct is " <<cT<<std::endl;
                        double x, y, z, qx, qy, qz, qw;
                        // std::string::size_type sz;     // alias of size_t

                        std::string str=transforms[i+j][2];
                        x = std::stod (str,nullptr);

                        str=transforms[i+j][3];
                        y = std::stod (str,nullptr);

                        str=transforms[i+j][4];
                        z = std::stod (str,nullptr);

                        str=transforms[i+j][5];
                        qx = std::stod (str,nullptr);

                        str=transforms[i+j][6];
                        qy = std::stod (str,nullptr);

                        str=transforms[i+j][7];
                        qz = std::stod (str,nullptr);

                        str=transforms[i+j][8];
                        qw = std::stod (str,nullptr);


                        // std::cout<<"return "<<qx<<" "<<qy<<" "<<qz<<" "<<qw<<std::endl;
                        Eigen::Quaterniond q(qw,qx,qy,qz);
                        transform(0,3)=x;
                        transform(1,3)=y;
                        transform(2,3)=z;
                        Eigen::Matrix3d qMatrix(q);
                        transform.block(0,0,3,3)=qMatrix;
                        // std::cout<<"return: \n"<<transform<<"\n"<<std::endl;
                        //std::cout<<"\n \n"<<std::endl;
                        // std::cout<<qMatrix<<endl;
                        if (invFlag) {return transform.inverse();}
                        else {return transform;}



                    }

                    j++;

                }


            }

        }
    }


    std::cout<<"csv lookup failed"<<std::endl;
    return Eigen::Matrix4d::Identity();
}

/*
void XmeansFn(std::vector<tf::Quaternion> num_test)
{
    std::default_random_engine rge;
    unsigned long seedint = time(NULL);
    // unsigned long seedint = 1612773364;
    std::cout << "Seed: " << seedint << std::endl;
    rge.seed(seedint);

    typedef XMeansT<DI,DataSetIterator,DataMetric,DataAggregator,CenterSplitter> XMeansQ;



    const unsigned int Nseed = 5;
    std::uniform_int_distribution<unsigned int> seedSel(0,Nseed-1);
    DI seed[Nseed];

    for (unsigned int k=0;k<Nseed;k++) {
        seed[k] = QRand(rge);
    }


    //for (size_t k=0;k<100;k++) {
    //int j = seedSel(rge);
    // num_test.push_back(dQRand(0.3,rge)*seed[j]);
    //}
    cout << "Number of instances: " << num_test.size() << endl;

    XMeansQ::CenterSet initial_centers;
    std::uniform_int_distribution<unsigned int> dataSel(0,99);
    initial_centers.push_back(num_test[dataSel(rge)]);
    initial_centers.push_back(num_test[dataSel(rge)]);

    cout << "running x-means " << endl;

    //minimum delta between points? not sure
    DataMetric metric;
    DataAggregator aggregator;
    CenterSplitter splitter;
    XMeansQ x(num_test.begin(), num_test.end(), initial_centers, N,
              metric, aggregator, splitter,
              10, 1e-3);

    x.process();

    const ClusterSet & clusters = x.clusters() ;
    unsigned int n_clusters = clusters.size();


    std::ofstream ocenters ("center", std::ofstream::out);
    std::ofstream odata ("data", std::ofstream::out);
    cout << "X-MEANS found " << n_clusters << " clusters." << endl;
    for(unsigned int i=0; i<n_clusters; i++){
        const Membership & c = clusters[i];
        ocenters << x.centers()[i] << " " << i << std::endl;
        cout << "CLUSTER "<<i<<" centered on " << x.centers()[i] << " has ";
        for(size_t j=0; j<c.size(); j++){
            if (j>0) {
                cout << ", " ;
            }
            const DI & D = num_test[c[j]];
            cout << "[" << D << "] ";
            odata << num_test[c[j]] << " " << i << std::endl;
        }
        cout << endl;
    }
    ocenters.close();
    odata.close();



}
*/
std::vector<std::vector<se3>> DmeansSe3Fn(std::vector<se3> num_test){

    std::default_random_engine rge;
    unsigned long seedint = time(NULL);
    // unsigned long seedint = 1612773364;
    std::cout << "Seed: " << seedint << std::endl;
    rge.seed(seedint);

    typedef DMeansT<DISE3,DataSetSe3Iterator,DataMetric,DataAggregator> DMeansQ;

    // DataSet num_test;
    const unsigned int Nseed = 5;
    std::uniform_int_distribution<unsigned int> seedSel(0,Nseed-1);
    DISE3 seed[Nseed];

    for (unsigned int k=0;k<Nseed;k++) {
        seed[k] = se3(VRand(rge),QRand(rge));
    }


    for (size_t k=0;k<100;k++) {
        int j = seedSel(rge);
        //     num_test.push_back(se3(dVRand(0.1,rge)+seed[j].t,dQRand(0.3,rge)*seed[j].q));
    }


    cout << "Number of instances: " << num_test.size() << endl;

    DMeansQ::CenterSet initial_centers;
    std::uniform_int_distribution<unsigned int> dataSel(0,99);
    initial_centers.push_back(num_test[dataSel(rge)]);
    initial_centers.push_back(num_test[dataSel(rge)]);

    cout << "running d-means " << endl;

    //minimum delta between points? not sure
    DataMetric metric;
    DataAggregator aggregator;
    DMeansQ x(num_test.begin(), num_test.end(), metric, aggregator, 2.0);

    x.process();

    const ClusterSet & clusters = x.clusters() ;
    unsigned int n_clusters = clusters.size();


    std::ofstream ocenters ("center", std::ofstream::out);
    std::ofstream odata ("data", std::ofstream::out);
    cout << "D-MEANS found " << n_clusters << " clusters." << endl;
    std::vector<std::vector<se3>> dataSe3;

    for(unsigned int i=0; i<n_clusters; i++){
        const Membership & c = clusters[i];
        ocenters << x.centers()[i] << " " << i << std::endl;
        cout << "CLUSTER "<<i<<" centered on " << x.centers()[i] << " has ";
        DataSetSe3 tempDatasetSe3;
        for(size_t j=0; j<c.size(); j++){
            if (j>0) {
                cout << ", " ;
            }
            const DISE3 & D = num_test[c[j]];

            tempDatasetSe3.push_back(num_test[c[j]]);
            cout << "[" << D << "] ";
            odata << num_test[c[j]] << " " << i << std::endl;
        }
        tempDatasetSe3.push_back(x.centers()[i]);
        dataSe3.push_back(tempDatasetSe3);
        cout << endl;
    }
    ocenters.close();
    odata.close();



    return dataSe3;
}

/*
void DmeansFn(std::vector<tf::Quaternion> num_test)
{
    std::default_random_engine rge;
    unsigned long seedint = time(NULL);
    // unsigned long seedint = 1612773364;
    //std::cout << "Seed: " << seedint << std::endl;
    rge.seed(seedint);

    typedef DMeansT<DI,DataSetIterator,DataMetric,DataAggregator> DMeansQ;



    const unsigned int Nseed = 5;
    std::uniform_int_distribution<unsigned int> seedSel(0,Nseed-1);
    DI seed[Nseed];

    for (unsigned int k=0;k<Nseed;k++) {
        seed[k] = QRand(rge);
    }


    //for (size_t k=0;k<100;k++) {
    //int j = seedSel(rge);
    // num_test.push_back(dQRand(0.3,rge)*seed[j]);
    //}
    cout << "Number of instances: " << num_test.size() << endl;

    DMeansQ::CenterSet initial_centers;
    std::uniform_int_distribution<unsigned int> dataSel(0,99);
    initial_centers.push_back(num_test[dataSel(rge)]);
    initial_centers.push_back(num_test[dataSel(rge)]);

    //cout << "running d-means " << endl;

    //minimum delta between points? not sure
    DataMetric metric;
    DataAggregator aggregator;
    CenterSplitter splitter;
    DMeansQ x(num_test.begin(),num_test.end(), metric, aggregator, 0.2);

    x.process();

    const ClusterSet & clusters = x.clusters() ;
    unsigned int n_clusters = clusters.size();


    std::ofstream ocenters ("center", std::ofstream::out);
    std::ofstream odata ("data", std::ofstream::out);
    cout << "D-MEANS found " << n_clusters << " clusters." << endl;
    for(unsigned int i=0; i<n_clusters; i++){
        const Membership & c = clusters[i];
        ocenters << x.centers()[i] << " " << i << std::endl;
        cout << "CLUSTER "<<i<<" centered on " << x.centers()[i] << " has ";
        for(size_t j=0; j<c.size(); j++){
            if (j>0) {
                cout << ", " ;
            }
            const DI & D = num_test[c[j]];
            cout << "[" << D << "] ";
            odata << num_test[c[j]] << " " << i << std::endl;
        }
        cout << endl;
    }
    ocenters.close();
    odata.close();



}
*/
int main(int argc, char *argv[]){



    //---------------------------------------------------------
    std::ifstream infile("transforms.csv");

    //double x, y, z, qx, qy, qz, qw;
    std::string KF, timePeriod, xStr, yStr, zStr, qxStr, qyStr, qzStr, qwStr;
    vector<vector<string>> transforms;
    while (infile >> KF)
    {

        //std::cout<<"KF "<<KF<<std::endl;
        std::string str=KF;
        vector<string> result;
        std::stringstream ss(KF);
        //stringstream ss=KF;
        while( ss.good() )
        {
            string substr;
            getline( ss, substr, ',' );
            result.push_back( substr );

        }
        transforms.push_back(result);
    }
    string maxKFStr=transforms.back()[0];
    std::string::size_type sz;   // alias of size_t

    unsigned int maxKF = std::stoi (maxKFStr,&sz);
    std::cout<<"maxKF is "<<maxKF<<std::endl;


    string maxTStr=transforms.back()[1];
    //  std::string::size_type sz;   // alias of size_t

    int maxT = std::stoi (maxTStr,&sz);
    std::cout<<"maxT is "<<maxT<<std::endl;



    //------------------------TEMPORAL ALIGNMENT--------------------------------


    int loopCount=0;
    std::vector<double> lowestT0Error;
    se3 defSe3({0,0,0},{0,0,0,1});
    double biggestError=5;
    //std::cout<<defSe3.q.x()<<" "<<defSe3.q.w()<<endl;
    //defSe3.t(0.0,0.0,0.0);
    //defSe3.q(0,0,0,1);
    vector<vector<vector<se3>>> tfVecData(maxT+1, vector<vector<se3>>(maxKF+1, vector<se3>(maxT+1, defSe3)));
    vector<vector<double>> errorVec(maxT+1, vector<double>(maxKF+1,-1));
    for (int t0=0; t0<=maxT; t0++){

        for(int i=0; i<=maxKF; i++){
            cout<<"this is KF "<<i<<endl;
            double Accum0=0;
            // double lowestT0Error=9999999999;
            //            double lowestT0=-1;
            double penaltyC=1;
            for (int j=0; j<=maxT; j++){

                DataSet localSet;
                DataSetSe3 localSe3Set;
                if (j==t0){continue;}

                int srcKF=i; int dstKF=i; int srcT=j; int dstT=t0;


                Eigen::Matrix4d T=parseData(srcKF, dstKF, srcT, dstT, transforms);
                // if (j<t0){T=T.inverse();}
                Eigen::Matrix3d Trot=T.block(0,0,3,3);
                Eigen::Quaterniond q(Trot);

                //std::cout<<T<<std::endl;

                tf::Quaternion quat(q.x(),q.y(),q.z(),q.w());
                tf::Vector3 t(T(0,3),T(1,3),T(2,3));
                se3 tf;
                tf.q=quat; tf.t=t;
                if (!T.isIdentity() ) {
                    // cout<<"first T is "<<T<<endl;
                    localSet.push_back(quat);
                    localSe3Set.push_back(tf);

                }
                else{

                    cout<< "ICP Failed" <<endl;
                }


                //   std::cout<<"0"<<std::endl;
                for (int t2=0; t2<=maxT; t2++){
                    if (t2==j ||t2==t0){continue;}
                    Eigen::Matrix4d T1, T2;

                    if (t2>t0 && t2>j ){
                        srcKF=i; dstKF=i; srcT=j; dstT=t2;
                        T1=parseData(srcKF, dstKF, srcT, dstT, transforms);
                        srcKF=i; dstKF=i; srcT=t2; dstT=t0;
                        T2=parseData(srcKF, dstKF, srcT, dstT, transforms);
                        //T=T2*T1.inverse();

                        //  std::cout<<"1"<<std::endl;

                    }

                    if (t2<t0 && t2<j){
                        srcKF=i; dstKF=i; srcT=j; dstT=t2;
                        T1=parseData(srcKF, dstKF, srcT, dstT, transforms);
                        srcKF=i; dstKF=i; srcT=t2; dstT=t0;
                        T2=parseData(srcKF, dstKF, srcT, dstT, transforms);

                        // std::cout<<"2"<<std::endl;

                    }

                    if (t2<j && t2>t0){
                        srcKF=i; dstKF=i; srcT=j; dstT=t2;
                        T1=parseData(srcKF, dstKF, srcT, dstT, transforms);
                        srcKF=i; dstKF=i; srcT=t2; dstT=t0;
                        T2=parseData(srcKF, dstKF, srcT, dstT, transforms);

                    }

                    if (t2>j && t2<t0){
                        srcKF=i; dstKF=i; srcT=j; dstT=t2;
                        T1=parseData(srcKF, dstKF, srcT, dstT, transforms);
                        srcKF=i; dstKF=i; srcT=t2; dstT=t0;
                        T2=parseData(srcKF, dstKF, srcT, dstT, transforms);
                    }
                    //T=T1.inverse()*T2.inverse();
                    if ( (T1.isIdentity(1e-3) )==1 || (T2.isIdentity(1e-1)==1 ) ){
                        cout<< "ICP Failed" <<endl;
                        // loopCount++;
                        continue;
                    }

                    T=T2*T1;
                    //std::cout<<T<<" ID? "<<T.isIdentity()<<std::endl;

                    Eigen::Matrix3d Trot=T.block(0,0,3,3);
                    Eigen::Quaterniond q(Trot);
                    tf::Quaternion quat(q.x(),q.y(),q.z(),q.w());
                    tf::Vector3 t(T(0,3),T(1,3),T(2,3));
                    se3 tf;
                    tf.q=quat; tf.t=t;
                    //  std::cout<<t.x()<<" "<<T<<std::endl;
                    localSet.push_back(quat);
                    localSe3Set.push_back(tf);
                    //  std::cout<<"4"<<std::endl;


                }


                /*     for (int Z=0; Z<localSet.size(); Z++)
                {
                    cout<<localSet[Z].x()<<","<<localSet[Z].y()<<","<<localSet[Z].z()<<","<<localSet[Z].w()<<endl;
                }
                */

                /*    if (loopCount==18){
                   cout<<"skipping run 18"<<endl;
                   loopCount++;
                   continue;
               }
            */

                std::cout<<"this is run #"<<loopCount<<std::endl;
                //XmeansFn(localSet);
                //DmeansFn(localSet);
                std::vector<std::vector<se3>> dataOut=DmeansSe3Fn(localSe3Set);
                std::cout<< dataOut.size()<<endl;
                if(localSe3Set.size()==dataOut.size()){    //1 sample per cluster->failure.
                    cout<<"couldn't verify connection"<<endl;
                    //penaltyC++;
                    Accum0=Accum0+10;
                }

                else{
                    double lowestError2=10;
                    unsigned int currentKk=999;
                    unsigned int cClusterSize;
                    unsigned int maxClusterSize=0;
                    for (int k=0; k<dataOut.size();k++){
                        cClusterSize=dataOut[k].size();
                        if (cClusterSize>maxClusterSize){
                            maxClusterSize=cClusterSize;
                        }

                    }

                    for (int k=0; k<dataOut.size();k++){
                        cClusterSize=dataOut[k].size();

                        if (cClusterSize<maxClusterSize)
                        {
                            continue;
                        }

                        //double xDst=dataOut[k].back().t.x();
                        //double yDst=dataOut[k].back().t.y();
                        //double zDst=dataOut[k].back().t.z();
                        tf::Vector3 tDst=dataOut[k].back().t;
                        tf::Quaternion qDst=dataOut[k].back().q;
                        //double qyDst=dataOut[k].back().q.y();
                        //double qzDst=dataOut[k].back().q.z();
                        //double qwDst=dataOut[k].back().q.w();
                        std::vector<tf::Vector3> tSrc;
                        std::vector<tf::Quaternion> qSrc;
                        if (dataOut[k].size()>1)
                        {
                            for (int kk=0; kk<dataOut[k].size()-1; kk++){
                                tSrc.push_back(dataOut[k][kk].t);
                                qSrc.push_back(dataOut[k][kk].q);

                            }
                            double errorTemp=0;
                            //tf::q
                            //tf::Quaternion q = q1 * q2.inverse();

                            for (int kkk=0; kkk<qSrc.size(); kkk++){
                                tf::Quaternion qq = qSrc[kkk] * qDst.inverse();
                                double cError= DataMetric::trans_weight*(tSrc[kkk]-tDst).length() + DataMetric::rot_weight*qq.getAngle();
                                //double tempDist=sqrt(pow(xSrc[kkk]-xDst,2)+pow(ySrc[kkk]-yDst,2)+pow(zSrc[kkk]-zDst,2));
                                errorTemp=errorTemp+cError;
                            }



                            double error=errorTemp/(dataOut[k].size()-1);
                            std::cout<<"cluster error is "<<error<<endl;





                            if (error<lowestError2){
                                lowestError2=error;
                                currentKk=k;
                            }




                        }



                    }

                    std::cout<<"chose cluster "<<currentKk<<endl;

                    Accum0=Accum0+lowestError2;

                    loopCount++;
                    std::cout<<"-----------\n"<<std::endl;
                    tfVecData[t0][i][j]=dataOut[currentKk].back();

                }

            }
            cout<<"Accum0 is "<<Accum0<<endl;
            if (Accum0>biggestError) {biggestError=Accum0;}
            Accum0=Accum0*penaltyC;
            if (Accum0==0){

                cout<<"still happening, KF is "<<i<<endl;
                //    Accum0=9999999;
            }
            cout<<"Accum0 is "<<Accum0<<endl;

            lowestT0Error.push_back(Accum0);
            errorVec[t0][i]=Accum0;
            /*  if (Accum0<lowestT0Error)
            {
                lowestT0Error=Accum0;
                lowestT0=t0;
                cout<<"best time period for KF "<<i<<" is "<<lowestT0<<endl;
            }
            */

        }



    }

    // for (unsigned int i=0; i<lowestT0Error.size();i++){
    //    cout<<"lowest T0 "<<lowestT0Error[i]<<endl;
    //    cout<<"lowest T0 size is "<<lowestT0Error.size()<<endl;
    //   cout<<"maxKF*maxT is"<<(maxKF+1)*(maxT+1)<<endl;


    //}
    std::vector<int> parentFrames;
    std::vector<std::vector<se3>> temporalTfs;
    for (unsigned int i=0; i<=maxKF;i++){
        double lowestError=99999999999;
        int selectedT=-1;
        for (unsigned int j=0; j<=maxT; j++){
            //  cout<<"index is "<<(i)*(maxT+1)+j<<endl;
            // double cError=lowestT0Error[(i)*(maxT+1)+j];
            double cError=errorVec[j][i];
            std::cout<<cError<<endl;
            if (cError<lowestError )
            {
                lowestError=cError;
                selectedT=j;
            }
        }
        std::vector<se3> temporalTfsTemp;
        parentFrames.push_back(selectedT);
        cout<<"best time period for KF "<<i<<" is "<<selectedT<<endl;   //time period match for every KF
        for (unsigned int j=0; j<=maxT; j++){
            if (j==selectedT){
                temporalTfsTemp.push_back(se3(tf::Vector3(0,0,0),tf::Quaternion::getIdentity()));
                continue;
            }
            tf::Quaternion qSelected=tfVecData[selectedT][i][j].q;
            tf::Vector3 tSelected=tfVecData[selectedT][i][j].t;
            temporalTfsTemp.push_back(se3(tSelected, qSelected));
            cout<<qSelected.x()<<" "<<qSelected.y()<<" "<<qSelected.z()<<" "<<qSelected.w()<<" "<<tSelected.x()<<" "<<tSelected.y()<<" "<<tSelected.z()<<endl;
        }

        temporalTfs.push_back(temporalTfsTemp);






    }


    //-----------------------------------spatial alignment-------------------------------------


    for(unsigned int i=1; i<=maxKF; i++){
        int t0=parentFrames[i];
        cout<<"this is Spatial KF "<<i<<endl;
        double Accum0=0;
        // double lowestT0Error=9999999999;
        //            double lowestT0=-1;
        DataSetSe3 localSe3Set;
        for (int j=0; j<=maxT; j++){



            //    if (j==t0){continue;}

            int srcKF=i; int dstKF=i-1; int srcT=j; int dstT=j;


            Eigen::Matrix4d T=parseData(srcKF, dstKF, srcT, dstT, transforms);
            if (T.isIdentity() ) {
                cout<<"ICP failed"<<endl;
                continue;
            }

            se3 currentTf=temporalTfs[i][j];
            se3 prevTf=temporalTfs[i-1][j];
            Eigen::Matrix4d prevT=Eigen::Matrix4d::Identity();
            Eigen::Matrix4d currentT=Eigen::Matrix4d::Identity();
            prevT(0,3)=prevTf.t.x();
            prevT(1,3)=prevTf.t.y();
            prevT(2,3)=prevTf.t.z();
            currentT(0,3)=currentTf.t.x();
            currentT(1,3)=currentTf.t.y();
            currentT(2,3)=currentTf.t.z();

            tf::Matrix3x3 mPrev(prevTf.q);
            Eigen::Matrix3d prevRot;
            tf::matrixTFToEigen(mPrev, prevRot);
            prevT.block(0,0,3,3)=prevRot;

            tf::Matrix3x3 mCurrent(currentTf.q);
            Eigen::Matrix3d currentRot;
            tf::matrixTFToEigen(mCurrent, currentRot);
            currentT.block(0,0,3,3)=currentRot;


            //currenT.block(0,0,3,3)=


            T=prevT*T*currentT.inverse();




            // if (j<t0){T=T.inverse();}
            Eigen::Matrix3d Trot=T.block(0,0,3,3);
            Eigen::Quaterniond q(Trot);

            //std::cout<<T<<std::endl;

            tf::Quaternion quat(q.x(),q.y(),q.z(),q.w());
            tf::Vector3 t(T(0,3),T(1,3),T(2,3));
            se3 tf;
            tf.q=quat; tf.t=t;
            if (!T.isIdentity() ) {
                // cout<<"first T is "<<T<<endl;

                localSe3Set.push_back(tf);

            }
            else{

                cout<< "ICP Failed" <<endl;
            }


        }
        std::vector<std::vector<se3>> dataOut=DmeansSe3Fn(localSe3Set);




    }





    return 0;
}



