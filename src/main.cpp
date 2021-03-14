#include <stdio.h>
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
#include "yaml-cpp/yaml.h"
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

double DataMetric::trans_weight = 1.0/0.3;   // was 1/0.3 default 1/0.15
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


// > means newer
unsigned int returnIndex(unsigned int i, unsigned int j, unsigned int maxKF, bool closeLoop){
    // maxKF starts at 0
    int idx= (maxKF+1) * j + i;
    closeLoop=false; //function changed, permanently disabled
    if (closeLoop)
    {
        idx= (maxKF-1+1) * j + (i-1);
        //    return (idx-1);
    }

    return (idx);
}




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


    if (dstT==srcT && dstKF>srcKF){
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
                //      std::cout<<"sKF is "<<sKF<<"dstKF  is "<<dstKF<<std::endl;
            if ((sKF==dstKF) && (sT==dstT)){

                unsigned int j=2;
                while (true){
                    if (transforms[i+j][0]=="%KF"){break;}

                    int cKF=stoi(transforms[i+j][0],nullptr,0);
                    int cT=stoi(transforms[i+j][1],nullptr,0);
                 //    std::cout<<"cT is "<<cT<<" dstT is "<<dstT<<std::endl;


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


std::vector<std::vector<se3>> dMeansSe3Fn(std::vector<se3> num_test){

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

se3 se3Inverse(se3 data){

    tf::Matrix3x3 m0(data.q);
    Eigen::Matrix3d Rot;
    Eigen::Matrix4d T =Eigen::Matrix4d::Identity();

    tf::matrixTFToEigen(m0, Rot);
    T.block(0,0,3,3)=Rot;

    T(0,3)=data.t.x();
    T(1,3)=data.t.y();
    T(2,3)=data.t.z();
    T=T.inverse().eval();

    se3 out;

    Rot=T.block(0,0,3,3);

    Eigen::Quaterniond q (Rot);

    out.q=tf::Quaternion(q.x(),q.y(),q.z(),q.w());
    out.t=tf::Vector3(T(0,3),T(1,3),T(2,3));


    return out;
}


se3 se3Mult(se3 data1, se3 data2){



    tf::Matrix3x3 m1(data1.q);

    Eigen::Matrix3d Rot1;
    Eigen::Matrix4d T1 =Eigen::Matrix4d::Identity();

    tf::matrixTFToEigen(m1, Rot1);
    T1.block(0,0,3,3)=Rot1;

    T1(0,3)=data1.t.x();
    T1(1,3)=data1.t.y();
    T1(2,3)=data1.t.z();



    tf::Matrix3x3 m2(data2.q);

    Eigen::Matrix3d Rot2;
    Eigen::Matrix4d T2 =Eigen::Matrix4d::Identity();

    tf::matrixTFToEigen(m2, Rot2);
    T2.block(0,0,3,3)=Rot2;

    T2(0,3)=data2.t.x();
    T2(1,3)=data2.t.y();
    T2(2,3)=data2.t.z();

    Eigen::Matrix4d T=T1*T2;


    se3 out;


    Rot1=T.block(0,0,3,3);

    Eigen::Quaterniond q(Rot1);

    out.q=tf::Quaternion(q.x(),q.y(),q.z(),q.w());
    out.t=tf::Vector3(T(0,3),T(1,3),T(2,3));

    return out;
}

int main(int argc, char *argv[]){

    float closeLoopUncertainty=1;

    YAML::Node config = YAML::LoadFile("../config.yaml");
    std::string loopCloseStr = config["closeLoop"].as<std::string>();

    bool closeLoop=false;

    if (loopCloseStr=="True" || loopCloseStr=="true")
    {
        closeLoop=true;
    }
    int sNode=0;



    std::ofstream output, graphInput, graphInitialEstimate;
    output.open("ol_transforms.csv");
    graphInput.open("graph_input.csv");
    graphInitialEstimate.open("graph_initial_estimate.csv");
    //---------------------------------------------------------
    std::ifstream infile("transforms.csv");

    //double x, y, z, qx, qy, qz, qw;
    std::string KF;
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


    //if (closeLoop){sNode=1; }

    //------------------------TEMPORAL ALIGNMENT--------------------------------


    int loopCount=0;
    std::vector<double> lowestT0Error;
    se3 defSe3({0,0,0},{0,0,0,1});
    double biggestError=5;
    //std::cout<<defSe3.q.x()<<" "<<defSe3.q.w()<<endl;
    //defSe3.t(0.0,0.0,0.0);
    //defSe3.q(0,0,0,1);
    vector<vector<vector<se3>>> tfVecData(maxT+1, vector<vector<se3>>(maxKF+1-sNode, vector<se3>(maxT+1, defSe3)));
    vector<vector<vector<double>>> temporalUncertainty(maxT+1, vector<vector<double>>(maxKF+1-sNode, vector<double>(maxT+1, 6)));
    vector<vector<double>> errorVec(maxT+1, vector<double>(maxKF+1-sNode,-1));
    for (int t0=0; t0<=maxT; t0++){

        for(int i=sNode; i<=maxKF; i++){
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
                   //  cout<<"first T is "<<T<<endl;
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

                std::cout<<"this is run #"<<loopCount<<std::endl;
                //XmeansFn(localSet);
                //DmeansFn(localSet);


                std::vector<std::vector<se3>> dataOut;
                float x=0.1;
                DataMetric::trans_weight = 1.0/x;
                DataMetric::rot_weight = 1.0/x;
                while(true){

                    dataOut=dMeansSe3Fn(localSe3Set);

                    if(localSe3Set.size()>dataOut.size() || x>1 )
                    {
                        break;
                    }
                    else
                    {
                        x=x+0.1;
                        DataMetric::trans_weight = 1.0/x;
                        DataMetric::rot_weight = 1.0/x;


                    }
                }

                // std::vector<std::vector<se3>> dataOut=dMeansSe3Fn(localSe3Set);

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
                    tfVecData[t0][i-sNode][j]=dataOut[currentKk].back();
                    temporalUncertainty[t0][i-sNode][j]=x;

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
            errorVec[t0][i-sNode]=Accum0;
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
    std::vector<std::vector<double>> temporalUncertaintyFiltered;

    for (unsigned int i=sNode; i<=maxKF;i++){
        double lowestError=99999999999;
        int selectedT=-1;
        for (unsigned int j=0; j<=maxT; j++){
            //  cout<<"index is "<<(i)*(maxT+1)+j<<endl;
            // double cError=lowestT0Error[(i)*(maxT+1)+j];
            double cError=errorVec[j][i-sNode];
            std::cout<<cError<<endl;
            if (cError<lowestError )
            {
                lowestError=cError;
                selectedT=j;
            }
        }
        std::vector<se3> temporalTfsTemp;
        std::vector<double> temporalUncertaintyFilteredTemp;
        parentFrames.push_back(selectedT);
        cout<<"best time period for KF "<<i<<" is "<<selectedT<<endl;   //time period match for every KF
        for (unsigned int j=0; j<=maxT; j++){
            if (j==selectedT){
                temporalTfsTemp.push_back(se3(tf::Vector3(0,0,0),tf::Quaternion::getIdentity()));
                temporalUncertaintyFilteredTemp.push_back(0.1);
                continue;
            }
            tf::Quaternion qSelected=tfVecData[selectedT][i-sNode][j].q;
            tf::Vector3 tSelected=tfVecData[selectedT][i-sNode][j].t;
            temporalTfsTemp.push_back(se3(tSelected, qSelected));
            temporalUncertaintyFilteredTemp.push_back(temporalUncertainty[selectedT][i-sNode][j]);
            cout<<qSelected.x()<<" "<<qSelected.y()<<" "<<qSelected.z()<<" "<<qSelected.w()<<" "<<tSelected.x()<<" "<<tSelected.y()<<" "<<tSelected.z()<<endl;
        }

        temporalTfs.push_back(temporalTfsTemp);

        temporalUncertaintyFiltered.push_back(temporalUncertaintyFilteredTemp);

    }
    //-----------------------------------spatial alignment-------------------------------------

    DataMetric::trans_weight = 1.0/0.1;   // was 1/0.3 default 1/0.15
    DataMetric::rot_weight = 1.0/0.1;

    vector<se3> spatialVecData(maxKF+1-sNode, defSe3);
    vector<double> uncertaintyCoeff(maxKF+1-sNode, 30);
    for(unsigned int i=sNode+1; i<=maxKF; i++){
        unsigned int t0=parentFrames[i-sNode];
        cout<<"this is Spatial KF "<<i<<endl;
        //double Accum0=0;
        // double lowestT0Error=9999999999;
        //            double lowestT0=-1;
        DataSetSe3 localSe3Set;
        for (unsigned int j=0; j<=maxT; j++){



            //    if (j==t0){continue;}

            int srcKF=i; int dstKF=i-1; int srcT=j; int dstT=j;

            Eigen::Matrix4d T=parseData(srcKF, dstKF, srcT, dstT, transforms);
            if (T.isIdentity() ) {
                cout<<"ICP failed"<<endl;
                continue;
            }

            se3 currentTf=temporalTfs[i-sNode][j];
            se3 prevTf=temporalTfs[i-1-sNode][j];
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

            if (  (prevT.isIdentity(1e-1) && parentFrames[i-sNode-1]!=j  ) || (currentT.isIdentity(1e-1) && j!= t0) ) {
                cout<<"ICP failed2"<<endl;
                continue;
            }
            Eigen::Matrix3d Ttemp=T.block(0,0,3,3);


            Eigen::Quaterniond qTemp(Ttemp);

            std::cout<<"KF is "<<i<<" q0 is "<<qTemp.x()<<" "<<qTemp.y()<<" "<<qTemp.z()<<" "<<qTemp.w()<<endl;



            T=prevT*T*(currentT.inverse());

            // cout<<"prev T is \n"<<prevT<<endl;
            //   cout<<"current T is \n"<<currentT<<endl;
            //   cout<<"currentT.inverse() is \n"<<currentT.inverse()<<endl;


            // if (j<t0){T=T.inverse();}
            Eigen::Matrix3d Trot=T.block(0,0,3,3);
            Eigen::Quaterniond q(Trot);



            std::cout<<"KF is "<<i<<" q1 is "<<q.x()<<" "<<q.y()<<" "<<q.z()<<" "<<q.w()<<endl;


            cout<<" ------------------------"<<endl;

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
        std::vector<std::vector<se3>> dataOut;
        float x=0.1;
        DataMetric::trans_weight = 1.0/x;
        DataMetric::rot_weight = 1.0/x;
        while(true){

            dataOut=dMeansSe3Fn(localSe3Set);

            if(localSe3Set.size()>dataOut.size() || x>1 )
            {
                break;
            }
            else
            {
                x=x+0.1;
                DataMetric::trans_weight = 1.0/x;
                DataMetric::rot_weight = 1.0/x;


            }
        }

        cout<<"metric is "<<x<<endl;


        if(localSe3Set.size()==dataOut.size()){    //1 sample per cluster->failure.
            cout<<"couldn't verify connection"<<endl;
            //penaltyC++;

        }

        else
        {
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
                tf::Vector3 tDst=dataOut[k].back().t;
                tf::Quaternion qDst=dataOut[k].back().q;
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
            spatialVecData[i-sNode]=dataOut[currentKk].back();
            uncertaintyCoeff[i-sNode]=x;



        }


    }




    /* //////////////////////////////////////////////////////////////////////////// */

    //Eigen::Matrix4d prevT=Eigen::Matrix4d::Identity();
    for (int i=0; i<spatialVecData.size(); i++){

        cout<<"chosen clusters "<<spatialVecData[i]<<endl;


    }

    //temporalTfs[KF][T] has the transforms to vertically align KFs
    //spatialVecData[KF] has horizontal spatial alignment
    std::vector<std::vector<se3>> olTransforms; //open loop transforms
    Eigen::Matrix4d incrementalSpatialTf =Eigen::Matrix4d::Identity();
    for (int i=sNode; i<=maxKF;i++){
        unsigned int t0=parentFrames[i-sNode];
        std::vector<se3> olTransformsTemp;
        //   cout<<"KF is "<<i<<endl;
        se3 spatialTQ=spatialVecData[i-sNode];

        tf::Matrix3x3 m(spatialTQ.q);

        Eigen::Matrix3d spatialRot;
        Eigen::Matrix4d spatialTf =Eigen::Matrix4d::Identity();

        tf::matrixTFToEigen(m, spatialRot);
        spatialTf.block(0,0,3,3)=spatialRot;

        spatialTf(0,3)=spatialTQ.t.x();
        spatialTf(1,3)=spatialTQ.t.y();
        spatialTf(2,3)=spatialTQ.t.z();


        incrementalSpatialTf=incrementalSpatialTf*spatialTf;


        //  cout<<"increment Sp TF is \n"<<endl;
        //   cout<< incrementalSpatialTf <<endl;

        //   cout<<"Sp TF is \n"<<endl;
        //   cout<<spatialTf<<endl;
        for (int j=0; j<=maxT; j++){

            se3 temporalTQ=temporalTfs[i-sNode][j];
            tf::Matrix3x3 m0(temporalTQ.q);

            Eigen::Matrix3d temporalRot;
            Eigen::Matrix4d temporalTf =Eigen::Matrix4d::Identity();

            tf::matrixTFToEigen(m0, temporalRot);
            temporalTf.block(0,0,3,3)=temporalRot;

            temporalTf(0,3)=temporalTQ.t.x();
            temporalTf(1,3)=temporalTQ.t.y();
            temporalTf(2,3)=temporalTQ.t.z();

            Eigen::Matrix4d T=incrementalSpatialTf*temporalTf;

            if (  temporalTf.isIdentity(1e-1) && t0!=j  ) {
                T=Eigen::Matrix4d::Identity();

            }
            //     std::cout<<temporalRot
            Eigen::Matrix3d rotT=T.block(0,0,3,3);
            Eigen::Quaterniond q(rotT);

            output<<i-sNode<<","<<j<<","<<T(0,3)<<","<<T(1,3)<<","<<T(2,3)<<","<<q.x()<<","<<q.y()<<","<<q.z()<<","<<q.w()<<","<<(j==t0)<<endl;

            se3 fPose;
            fPose.q=tf::Quaternion(q.x(),q.y(),q.z(),q.w());
            fPose.t=tf::Vector3(T(0,3),T(1,3),T(2,3));

            olTransformsTemp.push_back(fPose);

        }
        olTransforms.push_back(olTransformsTemp);

    }
    output.close();

    /* ------------------------------------FACTOR GRAPH SETUP--------------------------------------------------- */


    int maxNodes=(maxKF+1-sNode)*(maxT+1);
    graphInput<<std::fixed<<setprecision(20);
    for (int i=sNode; i<maxKF;i++){
        se3 L2=spatialVecData[i+1-sNode];

        unsigned int t0=parentFrames[i-sNode];
        unsigned int t1=parentFrames[i+1-sNode];
        double uncertainty=uncertaintyCoeff[i+1-sNode]; //because links to previous frame, here linking to forward frame
        for (int j=0; j<=maxT; j++){



            double temporalUncertaintyDouble = temporalUncertaintyFiltered[i-sNode][j];
            unsigned int serialIdx0=returnIndex(i,j,maxKF,closeLoop);  //S(Q0,t0)     //each kf is represented by a node tied to the node to the right and the node below it.
            unsigned int serialIdx1=returnIndex(i,t0,maxKF,closeLoop); //S(Q1,t0)
            unsigned int serialIdx2; //(SQ0,t1)

            se3 L1=temporalTfs[i-sNode][j];


            //se3 L1=se3Inverse(TQ0);
            // se3 L2=se3Inverse(TQ0);

            //  L1=se3Mult(L1,TQ1);
            //   L2=se3Mult(L2,TQ2);
            if (serialIdx0!=serialIdx1 ){

                //  se3 L1=se3Inverse(L1);
                graphInput << serialIdx1<<","<< serialIdx0<<","<< L1.t.x()<<","<< L1.t.y()<<","<<L1.t.z()<<","<<L1.q.x()<<","<< L1.q.y()<<","<<L1.q.z()<<","<< L1.q.w()<<","<<temporalUncertaintyDouble<<std::endl;   //from node, to node, x,y,z,qx,qy,qz,qw,uncertainty
            }

            if (j==t0){
                serialIdx2=returnIndex(i+1,t1,maxKF,closeLoop);
                graphInput << serialIdx0<<","<< serialIdx2<<","<< L2.t.x()<<","<< L2.t.y()<<","<<L2.t.z()<<","<<L2.q.x()<<","<< L2.q.y()<<","<<L2.q.z()<<","<< L2.q.w()<<","<<uncertainty  <<std::endl;   //from node, to node, x,y,z,qx,qy,qz,qw,uncertainty
            }
            if (i==maxKF-1 ){//handle spatial loop closure constraint

                t0=parentFrames[maxKF-sNode];
                temporalUncertaintyDouble = temporalUncertaintyFiltered[maxKF-sNode][j];
                serialIdx0=returnIndex(maxKF,j,maxKF,closeLoop);
                serialIdx1=returnIndex(maxKF,t0,maxKF,closeLoop);
                L1=temporalTfs[maxKF-sNode][j];
                if (serialIdx0!=serialIdx1 ){
                    //se3 L1=se3Inverse(L1);
                    graphInput << serialIdx1<<","<< serialIdx0<<","<< L1.t.x()<<","<< L1.t.y()<<","<<L1.t.z()<<","<<L1.q.x()<<","<< L1.q.y()<<","<<L1.q.z()<<","<< L1.q.w()<<","<<temporalUncertaintyDouble<<std::endl;   //from node, to node, x,y,z,qx,qy,qz,qw,uncertainty
                }


            }



        }

    }

    if(closeLoop){
        for (int j=0; j<=maxT; j++){


            bool found=false;

            unsigned int x;   //check which keyframes matches the special loop closure keyframe, usually the last keyframe.
            for (unsigned int i=0; i<transforms.size();i++){

                if (transforms[i][0]=="%KF"){
                    int sKF=stoi(transforms[i+1][0],nullptr,0);
                    int sT=stoi(transforms[i+1][1],nullptr,0);
                    if (sKF==-1 && sT==j){
                        int sKF2=stoi(transforms[i+2][0],nullptr,0);
                        int sT2=stoi(transforms[i+2][1],nullptr,0);

                        if (sKF2!=0 && sT2==j){
                            found=true;
                            x=sKF2;

                        }

                    }
                }

            }

            if (!found){continue;}


            //se3 T11=olTransforms[0][j];
            se3 T11=temporalTfs[0][j];
            Eigen::Matrix4d T111=Eigen::Matrix4d::Identity();
            Eigen::Quaterniond q11(T11.q.w(),T11.q.x(),T11.q.y(),T11.q.z() );
            Eigen::Matrix3d r11(q11);
            T111.block(0,0,3,3)=r11;
            T111(0,3)=T11.t.x();
            T111(1,3)=T11.t.y();
            T111(2,3)=T11.t.z();




            //se3 T22=olTransforms[x][j];
            se3 T22=temporalTfs[x][j];
            Eigen::Matrix4d T222=Eigen::Matrix4d::Identity();
            Eigen::Quaterniond q22(T22.q.w(),T22.q.x(),T22.q.y(),T22.q.z() );
            Eigen::Matrix3d r22(q22);
            T222.block(0,0,3,3)=r22;
            T222(0,3)=T22.t.x();
            T222(1,3)=T22.t.y();
            T222(2,3)=T22.t.z();


            std::cout<<"loop closure KF for time priod "<<j<<" is: "<<x<<std::endl;


            Eigen::Matrix4d T1=parseData(-1, 0, j, j, transforms);
            Eigen::Matrix4d T2=parseData(x, -1, j, j, transforms);


            unsigned int serialIdx1=returnIndex(0,j,maxKF,closeLoop); //S(Q1,t0)
            unsigned int serialIdx2=returnIndex(x,j,maxKF,closeLoop);  //S(Q0,t0)

            Eigen::Matrix4d T=T1*T2;


            T=  T111*T*T222.inverse();


            //T=T000*T*T.inverse();

            Eigen::Matrix3d R=T.block(0,0,3,3);

            Eigen::Quaterniond qd(R);
            se3 L1;
            L1.q=tf::Quaternion(qd.x(),qd.y(),qd.z(),qd.w());
            L1.t=tf::Vector3(T(0,3),T(1,3),T(2,3));
            if (serialIdx2!=serialIdx1){
                graphInput << serialIdx1<<","<< serialIdx2<<","<< L1.t.x()<<","<< L1.t.y()<<","<<L1.t.z()<<","<<L1.q.x()<<","<< L1.q.y()<<","<<L1.q.z()<<","<< L1.q.w()<<","<<closeLoopUncertainty <<std::endl;   //from node, to node, x,y,z,qx,qy,qz,qw,uncertainty
            }
        }
    }

    graphInput.close();
    graphInitialEstimate<<std::fixed<<setprecision(20);
    std::cout<<"max nodes is "<<maxNodes<<endl;
    std::default_random_engine generator;
    se3 prevT=olTransforms[sNode][0];
    const double mean = 0.0;
    for (int i=sNode; i<=maxKF;i++){

        for (int j=0; j<=maxT; j++){

            const double stddev = 0.2;
            const double translationCoeff=50;

            se3 T=olTransforms[i-sNode][j];
            Eigen::Quaterniond q0(T.q.w(),T.q.x(),T.q.y(),T.q.z());
            Eigen::Matrix3d rot(q0);
            Eigen::Matrix4d tMat=Eigen::Matrix4d::Identity();

            tMat(0,3)=T.t.x();
            tMat(1,3)=T.t.y();
            tMat(2,3)=T.t.z();

            if (i>sNode && tMat.isIdentity()){

                T=olTransforms[i-1-sNode][j];
                olTransforms[i-sNode][j]=T;

            }

            else{
                prevT=T;

            }

            int nodeIdx=returnIndex(i,j,maxKF,closeLoop); //S(Q1,t0)
            double x=T.t.x();
            double y=T.t.y();
            double z=T.t.z();
            double qx=T.q.x();
            double qy=T.q.y();
            double qz=T.q.z();
            double qw=T.q.w();

            if (i!=sNode){
                std::normal_distribution<double> dist(mean, stddev);

                x=x+dist(generator)*translationCoeff;
                y=y+dist(generator)*translationCoeff;
                z=z+dist(generator)*10;
                qx=qx+dist(generator);
                qy=qy+dist(generator);
                qz=qz+dist(generator);
                qw=qw+dist(generator);
            }
            tf::Quaternion q(qx, qy, qz, qw);

            q=q.normalize();

            graphInitialEstimate<<nodeIdx<<","<<x<<","<<y<<","<<z<<","<<q.x()<<","<<q.y()<<","<<q.z()<<","<<q.w()<<","<<i<<","<<j<<std::endl;

        }

    }
    graphInitialEstimate.close();
    return 0;
}
