#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;
using namespace std;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
  TODO:
    * Calculate the RMSE here.
  */
  int n = estimations[0].size();
  VectorXd rmse(n);
  rmse.fill(0);
  if(estimations.size() == 0 || estimations.size() != ground_truth.size()){
    cout << "Invalid estimation size" << endl;
    return rmse;
  }
  for(size_t i = 0; i < estimations.size(); i++){
    VectorXd residual;
    residual = estimations[i] - ground_truth[i];
    residual = residual.array() * residual.array();
    rmse += residual;
  }
  rmse /= estimations.size();
  rmse = rmse.array().sqrt();
  return rmse;
}