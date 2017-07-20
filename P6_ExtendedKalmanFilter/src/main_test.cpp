#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include "Eigen/Dense"
#include "FusionEKF.h"
#include "tools.h"

using namespace std;
using namespace Eigen;

int main()
{
    //Set measurements
    vector<MeasurementPackage> measurement_pack_list;
    string in_file_name_ = "../data/obj_pose-laser-radar-synthetic-input.txt";
    ifstream in_file(in_file_name_.c_str(), ifstream::in);
    if(!in_file.is_open()){
        cout << "Can not open the input file!" << endl;
    }
    string line;
    int i = 0;
    while(getline(in_file, line) && (i <= 6)){
        MeasurementPackage meas_package;
        istringstream iss(line);
        string sensor_type;
        iss >> sensor_type;
        cout << sensor_type << endl;
        long timestamp;
        if(sensor_type.compare("R") == 0){
            meas_package.sensor_type_ = MeasurementPackage::RADAR;
            float x, y, z;
            iss >> x;
            iss >> y;
            iss >> z;
            meas_package.raw_measurements_ = VectorXd(3);
            meas_package.raw_measurements_ << x, y, z;
            iss >> timestamp;
            meas_package.timestamp_ = timestamp;
            measurement_pack_list.push_back(meas_package); 
        }
        else if(sensor_type.compare("L") == 0){
            meas_package.sensor_type_ = MeasurementPackage::LASER;
            float x, y;
            iss >> x;
            iss >> y;
            meas_package.raw_measurements_ = VectorXd(2);
            meas_package.raw_measurements_ << x, y;
            iss >> timestamp;
            meas_package.timestamp_ = timestamp;
            measurement_pack_list.push_back(meas_package);
        }
        i++;
    }
    FusionEKF f;
    size_t N = measurement_pack_list.size();
    for(size_t k = 0; k < N; k++){
        f.ProcessMeasurement(measurement_pack_list[k]);
    }
    if(in_file.is_open()){
        in_file.close();
    }
    return 0;
}