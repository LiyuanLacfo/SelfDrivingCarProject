#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include "ukf.h"
// #include "ukf_1.h"
#include "tools.cpp"
#include "Eigen/Dense"
#include "ground_truth.h"

using namespace Eigen;

using namespace std;

int main(int argc, char * argv[])
{
    vector<MeasurementPackage> measurement_pack_list;
    vector<GroundTruthPackage> gt_pack_list;
    string in_file_name = argv[1];
    string out_file_name = argv[2];
    ifstream in_file(in_file_name, ifstream::in);
    ofstream out_file(out_file_name, ofstream::out);
    if(!in_file.is_open()){
        cout << "The file can't be opened" << endl;
    }
    string line;
    while(getline(in_file, line)){
        MeasurementPackage meas_pack;
        GroundTruthPackage gt_pack;
        istringstream iss(line);
        string sensor_type;
        iss >> sensor_type;
        // cout << sensor_type << endl;
        long time_stamp;
        if(sensor_type.compare("L") == 0){
            meas_pack.sensor_type_ = MeasurementPackage::LASER;
            float x, y;
            iss >> x;
            iss >> y;
            meas_pack.raw_measurements_ = VectorXd(2);
            meas_pack.raw_measurements_ << x, y;
            iss >> time_stamp;
            meas_pack.timestamp_ = time_stamp;
            measurement_pack_list.push_back(meas_pack);
        }
        else if(sensor_type.compare("R") == 0){
            meas_pack.sensor_type_ = MeasurementPackage::RADAR;
            float x, y, z;
            iss >> x;
            iss >> y;
            iss >> z;
            meas_pack.raw_measurements_ = VectorXd(3);
            meas_pack.raw_measurements_ << x, y, z;
            iss >> time_stamp;
            meas_pack.timestamp_ = time_stamp;
            measurement_pack_list.push_back(meas_pack);
        }
        //read ground truth value
        double x_gt;
        double y_gt;
        double vx_gt;
        double vy_gt;
        iss >> x_gt;
        iss >> y_gt;
        iss >> vx_gt;
        iss >> vy_gt;
        gt_pack.gt_values_ = VectorXd(4);
        gt_pack.gt_values_ << x_gt, y_gt, vx_gt, vy_gt;
        gt_pack_list.push_back(gt_pack);
    }
    vector<VectorXd> estimations;
    vector<VectorXd> groundtruths;

    UKF ukf;
    
    out_file << "time_stamp" << "\t";
    out_file << "px_state" << "\t";
    out_file << "py_state" << "\t";
    out_file << "v_state" << "\t";
    out_file << "yaw_state" << "\t";
    out_file << "yaw_rate_state" << "\t";
    out_file << "sensor type" << "\t";
    out_file << "NIS" << "\n";

    for(size_t i = 0; i < measurement_pack_list.size(); i++){
        ukf.ProcessMeasurement(measurement_pack_list[i]);
        out_file << measurement_pack_list[i].timestamp_ << "\t";
        out_file << ukf.x_(0) << "\t";
        out_file << ukf.x_(1) << "\t";
        out_file << ukf.x_(2) << "\t";
        out_file << ukf.x_(3) << "\t";
        out_file << ukf.x_(4) << "\t";
        out_file << measurement_pack_list[i].sensor_type_ << "\t";
        out_file << (measurement_pack_list[i].sensor_type_ == MeasurementPackage::LASER ? ukf.NIS_Laser_ : ukf.NIS_Radar_) << "\n";
        //estimation
        double x_estimation = ukf.x_(0);
        double y_estimation = ukf.x_(1);
        double vx_estimation = ukf.x_(2)*cos(ukf.x_(3));
        double vy_estimation = ukf.x_(2)*sin(ukf.x_(3));
        VectorXd cur_estimation(4);
        cur_estimation << x_estimation, y_estimation, vx_estimation, vy_estimation;
        estimations.push_back(cur_estimation);
        groundtruths.push_back(gt_pack_list[i].gt_values_);
    }
    // compute the accuracy (RMSE)
    Tools tools;
    VectorXd rmse = tools.CalculateRMSE(estimations, groundtruths);
    cout << "Accuracy - RMSE:" << endl << rmse << endl;

    // save accuracy value
    ofstream out_file_accuracy_("accuracy.txt", ofstream::out);
    out_file_accuracy_ << "Accuracy - RMSE: " << endl << rmse << endl;

    // close open files
    if (out_file.is_open())           out_file.close();
    if (in_file.is_open())            in_file.close();
    if (out_file_accuracy_.is_open())  out_file_accuracy_.close();
    return 0;
}