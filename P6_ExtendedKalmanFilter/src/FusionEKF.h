#ifndef FusionEKF_H_
#define FusionEKF_H_

#include "measurement_package.h"
#include "Eigen/Dense"
#include <vector>
#include <string>
#include <fstream>
#include "kalman_filter.h"

#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;
#include "tools.h"

class FusionEKF {
public:
  /**
  * Constructor.
  */
  FusionEKF();

  /**
  * Destructor.
  */
  virtual ~FusionEKF();

  /**
  * Run the whole flow of the Kalman Filter from here.
  */
  void ProcessMeasurement(const MeasurementPackage &measurement_pack);

  /**
  * Kalman Filter update and prediction math lives in here.
  */
  KalmanFilter ekf_;

private:
  // check whether the tracking toolbox was initialized or not (first measurement)
  bool is_initialized_;

  // previous timestamp
  long long previous_timestamp_;

  float noise_ax_;

  float noise_ay_;

  // tool object used to compute Jacobian and RMSE
  Tools tools;
  Eigen::MatrixXd R_laser_;
  Eigen::MatrixXd R_radar_;
  Eigen::MatrixXd H_laser_;
  Eigen::MatrixXd Hj_;
};

// FusionEKF::FusionEKF() {
//   is_initialized_ = false;

//   previous_timestamp_ = 0;

//   // initializing matrices
//   R_laser_ = MatrixXd(2, 2);
//   R_radar_ = MatrixXd(3, 3);
//   H_laser_ = MatrixXd(2, 4);
//   Hj_ = MatrixXd(3, 4);

//   //measurement covariance matrix - laser
//   R_laser_ << 0.0225, 0,
//               0, 0.0225;

//   //measurement covariance matrix - radar
//   R_radar_ << 0.09, 0, 0,
//               0, 0.0009, 0,
//               0, 0, 0.09;
//   H_laser_ << 1, 0, 0, 0,
//               0, 1, 0, 0;

//   Hj_ << 0, 0, 0, 0,
//          0, 0, 0, 0,
//          0, 0, 0, 0;

//   noise_ax_ = 9;
//   noise_ay_ = 9;

//   ekf_.ax_ = noise_ax_;
//   ekf_.ay_ = noise_ay_;
//   ekf_.P_ = MatrixXd(4, 4);
//   ekf_.P_ << 1, 0, 0, 0,
//               0, 1, 0, 0, 
//               0, 0, 1000, 0,
//               0, 0, 0, 1000;

//   /**
//   TODO:
//     * Finish initializing the FusionEKF.
//     * Set the process and measurement noises
//   */


// }

// /**
// * Destructor.
// */
// FusionEKF::~FusionEKF() {}

// void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


//   /*****************************************************************************
//    *  Initialization
//    ****************************************************************************/
//   if (!is_initialized_) {
//     /**
//     TODO:
//       * Initialize the state ekf_.x_ with the first measurement.
//       * Create the covariance matrix.
//       * Remember: you'll need to convert radar from polar to cartesian coordinates.
//     */
//     // first measurement
//     cout << "EKF: " << endl;
//     ekf_.x_ = VectorXd(4);
//     ekf_.x_ << 1, 1, 1, 1;

//     if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
//       /**
//       Convert radar from polar to cartesian coordinates and initialize state.
//       */
//       float rho = measurement_pack.raw_measurements_(0);
//       float phi = measurement_pack.raw_measurements_(1);
//       float rho_dot = measurement_pack.raw_measurements_(2);
//       float px = rho*cos(phi);
//       float py = rho*sin(phi);
//       float vx = rho_dot*cos(phi);
//       float vy = rho_dot*sin(phi);
//       ekf_.x_ << px, py, vx, vy;
//     }
//     else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
//       /**
//       Initialize state.
//       */
//       ekf_.x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0;
//     }

//     // done initializing, no need to predict or update
//     previous_timestamp_ = measurement_pack.timestamp_;
//     is_initialized_ = true;
//     return;
//   }


//   /*****************************************************************************
//    *  Prediction
//    ****************************************************************************/

//   /**
//    TODO:
//      * Update the state transition matrix F according to the new elapsed time.
//       - Time is measured in seconds.
//      * Update the process noise covariance matrix.
//      * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
//    */
//   long long cur_timestamp = measurement_pack.timestamp_;
//   // cout << "cur: " << cur_timestamp << endl;
//   // cout << "previous: " << previous_timestamp_ << endl;
//   // cout << cur_timestamp - previous_timestamp_ << endl;
//   double dt = (cur_timestamp - previous_timestamp_)/1000000.0;
//   // cout << "dt: " << dt << endl;
//   previous_timestamp_ = cur_timestamp;
//   ekf_.UpdateF(dt);
//   // cout << "F: " << ekf_.F_ << endl;
//   ekf_.UpdateQ(dt);
//   // cout << "Q: " << ekf_.Q_ << endl;
//   ekf_.Predict();
//   // cout << "x: " << ekf_.x_ << endl;

//   /*****************************************************************************
//    *  Update
//    ****************************************************************************/

//   /**
//    TODO:
//      * Use the sensor type to perform the update step.
//      * Update the state and covariance matrices.
//    */

//   if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
//     // Radar updates
//     ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
//     ekf_.R_ = R_radar_;
//     ekf_.UpdateEKF(measurement_pack.raw_measurements_);
//   } else {
//     // Laser updates
//     ekf_.H_ = H_laser_;
//     ekf_.R_ = R_laser_;
//     ekf_.Update(measurement_pack.raw_measurements_);
//   }

//   // print the output
//   cout << "x_ = " << ekf_.x_ << endl;
//   cout << "P_ = " << ekf_.P_ << endl;
// }


#endif /* FusionEKF_H_ */
