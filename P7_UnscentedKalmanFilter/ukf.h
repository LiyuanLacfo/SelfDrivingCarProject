#ifndef UKF_H
#define UKF_H

#include "measurement_package.h"
#include "Eigen/Dense"
#include <vector>
#include <string>
#include <fstream>
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;
using namespace std;

class UKF {
public:

  ///* initially set to false, set to true in first call of ProcessMeasurement
  bool is_initialized_;

  ///* if this is false, laser measurements will be ignored (except for init)
  bool use_laser_;

  ///* if this is false, radar measurements will be ignored (except for init)
  bool use_radar_;

  ///* state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
  VectorXd x_;

  ///* state covariance matrix
  MatrixXd P_;


  ///* time when the state is true, in us
  long long time_us_;

  ///* Process noise standard deviation longitudinal acceleration in m/s^2
  double std_a_;

  ///* Process noise standard deviation yaw acceleration in rad/s^2
  double std_yawdd_;

  ///* Laser measurement noise standard deviation position1 in m
  double std_laspx_;

  ///* Laser measurement noise standard deviation position2 in m
  double std_laspy_;

  ///* Radar measurement noise standard deviation radius in m
  double std_radr_;

  ///* Radar measurement noise standard deviation angle in rad
  double std_radphi_;

  ///* Radar measurement noise standard deviation radius change in m/s
  double std_radrd_ ;

  ///* Weights of sigma points
  VectorXd weights_;

  ///* State dimension
  int n_x_;

  ///* Augmented state dimension
  int n_aug_;

  ///* Sigma point spreading parameter
  double lambda_;
  //NIS of laser
  double NIS_Laser_;
  //NIS of radar
  double NIS_Radar_;

  //Augmented sigma points matrix
  MatrixXd AugSigmaPoints;
  //Prediction augmented sigma points matrix
  MatrixXd PredAugSigmaPoints;
  //Laser Measurement covariance matrix
  MatrixXd R_Laser_;
  //Radar Measurement covariance matrix
  MatrixXd R_Radar_;
  //Radar Measurement prediction
  VectorXd PredRadar;
  //Radar Measurement prediction covariance matrix
  MatrixXd PredRadarCovariance;
  //Radar Sigma Points
  MatrixXd RadarSigmaPoints;
  


  /**
   * Constructor
   */
  UKF();

  /**
   * Destructor
   */
  virtual ~UKF();

  /**
   * ProcessMeasurement
   * @param meas_package The latest measurement data of either radar or laser
   */
  void ProcessMeasurement(MeasurementPackage meas_package);

  /**
   * Prediction Predicts sigma points, the state, and the state covariance
   * matrix
   * @param delta_t Time between k and k+1 in s
   */
  void Prediction(double delta_t);

  /**
   * Updates the state and the state covariance matrix using a laser measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateLidar(MeasurementPackage meas_package);

  /**
   * Updates the state and the state covariance matrix using a radar measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateRadar(MeasurementPackage meas_package);
  void GenerateAugSigmaPoints();
  void PredictAugSigmaPoints(double delta_t);
  void PredictMeanAndCovariance();
  void FirstMeasurement(MeasurementPackage meas_package);
  void LaserUpdate(MeasurementPackage meas_package);
  void PredictRadarMeasurement();
  void RadarUpdate(MeasurementPackage meas_package);
};


/**
 * Initializes Unscented Kalman filter
 */
// UKF::UKF() {
//   // if this is false, laser measurements will be ignored (except during init)
//   use_laser_ = true;

//   // if this is false, radar measurements will be ignored (except during init)
//   use_radar_ = true;
//   // set is_initialized_ true
//   is_initialized_ = false;

//   n_x_ = 5;//the number of state variable
//   n_aug_ = 7;//the number of augmented state variable

//   // initial state vector
//   x_ = VectorXd(n_x_);
//   x_ << 1, 1, 1, 1, 1;

//   // initial covariance matrix
//   P_ = MatrixXd(n_x_, n_x_);
//   P_ = MatrixXd::Identity(n_x_, n_x_);

//   // Process noise standard deviation longitudinal acceleration in m/s^2
//   std_a_ = 1;

//   // Process noise standard deviation yaw acceleration in rad/s^2
//   std_yawdd_ = 1;

//   // Laser measurement noise standard deviation position1 in m
//   std_laspx_ = 0.15;

//   // Laser measurement noise standard deviation position2 in m
//   std_laspy_ = 0.15;

//   // Radar measurement noise standard deviation radius in m
//   std_radr_ = 0.3;

//   // Radar measurement noise standard deviation angle in rad
//   std_radphi_ = 0.03;

//   // Radar measurement noise standard deviation radius change in m/s
//   std_radrd_ = 0.3;

  
//   lambda_ = 3 - n_aug_; //the design parameter for sigma points
//   weights_ = VectorXd(2*n_aug_+1);
//   //Laser Measurement covariance matrix
//   R_Laser_ = MatrixXd(2, 2);
//   R_Laser_ << std_laspx_ * std_laspx_, 0,
//               0, std_laspy_ * std_laspy_;
//   //Radar Measurement covariance matrix
//   R_Radar_ = MatrixXd(3, 3);
//   R_Radar_ << std_radr_ * std_radr_, 0, 0,
//               0, std_radphi_ * std_radphi_, 0,
//               0, 0, std_radrd_ * std_radrd_;
//   //fill weights
//   weights_ = VectorXd(2*n_aug_+1);
//   weights_(0) = lambda_ / (lambda_+n_aug_);
//   for(int i = 1; i < 2*n_aug_+1; i++){
//     weights_(i) = 1.0/(2*(lambda_+n_aug_));
//   }
// }

// UKF::~UKF() {}

// /**
//  * @param {MeasurementPackage} meas_package The latest measurement data of
//  * either radar or laser.
//  */
// void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
//   /**
//   TODO:

//   Complete this function! Make sure you switch between lidar and radar
//   measurements.
//   */
//   if(!is_initialized_){
//     FirstMeasurement(meas_package);
//     return;
//   }
//   long long cur_timestamp = meas_package.timestamp_;
//   double delta_t = (cur_timestamp - time_us_) / 1000000.0;
//   time_us_ = cur_timestamp;
//   //Prediction based on the previous state
//   Prediction(delta_t);
//   // update based on the sensor type
//   if(meas_package.sensor_type_ == MeasurementPackage::LASER){
//     LaserUpdate(meas_package);
//   }
//   else{
//     RadarUpdate(meas_package);
//   }
// }

// /**
//  * Predicts sigma points, the state, and the state covariance matrix.
//  * @param {double} delta_t the change in time (in seconds) between the last
//  * measurement and this one.
//  */
// void UKF::Prediction(double delta_t) {
//   /**
//   TODO:

//   Complete this function! Estimate the object's location. Modify the state
//   vector, x_. Predict sigma points, the state, and the state covariance matrix.
//   */

//   //generate sigma points
//   GenerateAugSigmaPoints();
//   //prediction sigma points
//   PredictAugSigmaPoints(delta_t);
//   //Predict mean state variable and covariance
//   PredictMeanAndCovariance();
// }

// /**
//  * Updates the state and the state covariance matrix using a laser measurement.
//  * @param {MeasurementPackage} meas_package
//  */
// void UKF::UpdateLidar(MeasurementPackage meas_package) {
//   /**
//   TODO:

//   Complete this function! Use lidar data to update the belief about the object's
//   position. Modify the state vector, x_, and covariance, P_.

//   You'll also need to calculate the lidar NIS.
//   */
// }

// /**
//  * Updates the state and the state covariance matrix using a radar measurement.
//  * @param {MeasurementPackage} meas_package
//  */
// void UKF::UpdateRadar(MeasurementPackage meas_package) {}
//   /**
//   TODO:

//   Complete this function! Use radar data to update the belief about the object's
//   position. Modify the state vector, x_, and covariance, P_.

//   You'll also need to calculate the radar NIS.
//   */

// void UKF::GenerateAugSigmaPoints()
// {
//   AugSigmaPoints = MatrixXd(n_aug_, 2*n_aug_+1);
//   //augmented x vector where the first 5 are state vector, the last 2 are accerleration and yaw accerleration
//   VectorXd x_aug(n_aug_);
//   x_aug.head(n_x_) = x_;
//   x_aug(n_x_) = 0;
//   x_aug(n_x_+1) = 0;
//   //the covariance matrix of accerleration and yaw accerleration
//   MatrixXd Q(n_aug_ - n_x_, n_aug_ - n_x_);
//   Q << std_a_*std_a_, 0,
//        0, std_yawdd_*std_yawdd_;
//   // augmented covariance matrix
//   MatrixXd P_aug(n_aug_, n_aug_);
//   P_aug.fill(0.0);
//   P_aug.topLeftCorner(n_x_, n_x_) = P_;
//   P_aug.bottomRightCorner(n_aug_ - n_x_, n_aug_ - n_x_) = Q;
//   // the cholesky factorization matrix of P_aug
//   MatrixXd A = P_aug.llt().matrixL();
//   //fill AugSigmaPoint
//   AugSigmaPoints.col(0) = x_aug;
//   for(int i = 0; i < n_aug_; i++){
//     AugSigmaPoints.col(i+1) = x_aug + sqrt(lambda_+n_aug_)*A.col(i);
//     // while(AugSigmaPoints.col(i+1)(3) > M_PI) AugSigmaPoints.col(i+1)(3) -= 2*M_PI;
//     // while(AugSigmaPoints.col(i+1)(3) < -M_PI) AugSigmaPoints.col(i+1)(3) += 2*M_PI;
//     AugSigmaPoints.col(i+n_aug_+1) = x_aug - sqrt(lambda_+n_aug_)*A.col(i);
//     // while(AugSigmaPoints.col(i+1+n_aug_)(3) > M_PI) AugSigmaPoints.col(i+1+n_aug_)(3) -= 2*M_PI;
//     // while(AugSigmaPoints.col(i+1+n_aug_)(3) < -M_PI) AugSigmaPoints.col(i+1+n_aug_)(3) += 2*M_PI;
//   }
// }

// void UKF::PredictAugSigmaPoints(double delta_t)
// {
//   PredAugSigmaPoints = MatrixXd(n_x_, 2*n_aug_+1);
//   double eps = 0.001;
//   for(int i = 0; i < 2*n_aug_+1; i++){
//     double v = AugSigmaPoints.col(i)[2];
//     double phi = AugSigmaPoints.col(i)[3];
//     double phi_dot = AugSigmaPoints.col(i)[4];
//     double nu_a = AugSigmaPoints.col(i)[5];
//     double nu_phi_double_dot = AugSigmaPoints.col(i)[6];
//     VectorXd e1(n_x_);//the delta part during delta_t 
//     VectorXd e2(n_x_);//the noise part during delta_t
//     if(fabs(phi_dot) < eps){
//       e1(0) = v*cos(phi)*delta_t;
//       e1(1) = v*sin(phi)*delta_t;
//       e1.tail(3) << 0, 0, 0;
//     }
//     else{
//       e1(0) = v/phi_dot*(sin(phi+phi_dot*delta_t) - sin(phi));
//       e1(1) = v/phi_dot*(-cos(phi+phi_dot*delta_t) + cos(phi));
//       e1(2) = 0;
//       e1(3) = phi_dot*delta_t;
//       e1(4) = 0;
//     }
//     e2(0) = 0.5*delta_t*delta_t*cos(phi)*nu_a;
//     e2(1) = 0.5*delta_t*delta_t*sin(phi)*nu_a;
//     e2(2) = delta_t*nu_a;
//     e2(3) = 0.5*delta_t*delta_t*nu_phi_double_dot;
//     e2(4) = nu_phi_double_dot*delta_t;
//     //sanity check
//     double px = AugSigmaPoints.col(i)[0];
//     double py = AugSigmaPoints.col(i)[1];
//     if(fabs(px) < 0.001 && fabs(px) < 0.001){
//       AugSigmaPoints.col(i)[0] = 0.1;
//       AugSigmaPoints.col(i)[1] = 0.1;
//     }
//     PredAugSigmaPoints.col(i) = AugSigmaPoints.col(i).head(n_x_) + e1 + e2;
//     //normalize phi
//     // while(PredAugSigmaPoints.col(i)(3) > M_PI) PredAugSigmaPoints.col(i)(3) -= 2*M_PI;
//     // while(PredAugSigmaPoints.col(i)(3) < -M_PI) PredAugSigmaPoints.col(i)(3) += 2*M_PI;
//   }
// }

// void UKF::PredictMeanAndCovariance(){
//   //predict mean state variable
//   x_.fill(0.0);
//   for(int i = 0; i < 2*n_aug_+1; i++){
//     x_ += weights_(i)*PredAugSigmaPoints.col(i);
//   }
//   //normalize x
//   // while(x_(3) > M_PI) x_(3) -= 2*M_PI;
//   // while(x_(3) < -M_PI) x_(3) += 2*M_PI;
//   //predict covariance matrix
//   P_.fill(0.0);
//   for(int i = 0; i < 2*n_aug_+1; i++){
//     VectorXd diff = PredAugSigmaPoints.col(i) - x_;
//     //normalize diff
//     while(diff(3) > M_PI) diff(3) -= 2*M_PI;
//     while(diff(3) < -M_PI) diff(3) += 2*M_PI;
//     P_ += weights_(i) * diff * diff.transpose();
//   }
// }

// void UKF::FirstMeasurement(MeasurementPackage meas_package)
// {
//   time_us_ = meas_package.timestamp_;
//   if(meas_package.sensor_type_ == MeasurementPackage::LASER){
//     x_.fill(0.0);
//     double x, y;
//     x = meas_package.raw_measurements_(0);
//     y = meas_package.raw_measurements_(1);
//     x_(0) = x;
//     x_(1) = y;
//   }
//   else{
//     x_.fill(0);
//     double rho = meas_package.raw_measurements_(0);
//     double phi = meas_package.raw_measurements_(1);
//     double rho_dot = meas_package.raw_measurements_(2);
//     x_(0) = rho*cos(phi);
//     x_(1) = rho*sin(phi);
//   }
//   is_initialized_ = true;
// }

// void UKF::LaserUpdate(MeasurementPackage meas_package)
// {
//   int n_z = 2;
//   MatrixXd Zsig = PredAugSigmaPoints.block(0, 0, n_z, 2*n_aug_+1);
//   VectorXd z_pred(n_z);
//   z_pred.fill(0.0);
//   for(int i = 0; i < 2*n_aug_+1; i++){
//     z_pred += weights_(i)*Zsig.col(i);
//   }
//   MatrixXd S(n_z, n_z); // the covariance matrix of radar prediciton measurement
//   //fill S
//   S.fill(0.0);
//   for(int i = 0; i < 2*n_aug_+1; i++){
//     VectorXd diff = Zsig.col(i) - z_pred;
//     S += weights_(i)*diff*diff.transpose();
//   }
//   S += R_Laser_;
//   VectorXd PredLidar = VectorXd(n_z);
//   PredLidar = z_pred;
//   MatrixXd PredLidarCovariance = MatrixXd(n_z, n_z);
//   PredLidarCovariance = S;
//   //Update
//   MatrixXd T(n_x_, n_z);
//   T.fill(0.0);
//   for(int i = 0; i < 2 * n_aug_+1; i++){
//     VectorXd diff_x = PredAugSigmaPoints.col(i) - x_;
//     VectorXd diff_z = Zsig.col(i) - PredLidar;
//     T += weights_(i) * diff_x * diff_z.transpose();
//   }
//   //Kalman gain
//   MatrixXd K = T*PredLidarCovariance.inverse();
//   //update state mean vector
//   x_ = x_ + K*(meas_package.raw_measurements_ - PredLidar);
//   //update state covariance
//   P_ -= K*PredLidarCovariance*K.transpose();
//   NIS_Laser_ = (meas_package.raw_measurements_ - z_pred).transpose() * S.inverse() *
//         (meas_package.raw_measurements_ - z_pred);


//   // MatrixXd H(2, n_x_);
//   // H << 1, 0, 0, 0, 0,
//   //      0, 1, 0, 0, 0;
//   // VectorXd z = H*x_;
//   // VectorXd y = meas_package.raw_measurements_ - z;
//   // MatrixXd S = H*P_*H.transpose() + R_Laser_;
//   // MatrixXd K = P_*H.transpose()*S.inverse();
//   // //Update mean state vector
//   // x_ = x_ + K*y;
//   // //update state covariance matrix
//   // MatrixXd I = MatrixXd::Identity(n_x_, n_x_);
//   // P_ = (I - K*H)*P_;
// }

// void UKF::PredictRadarMeasurement()
// {
//   int n_z = 3;
//   RadarSigmaPoints = MatrixXd(n_z, 2*n_aug_+1);
//   for(int i = 0; i < 2*n_aug_+1; i++){
//     double px = PredAugSigmaPoints.col(i)[0];
//     double py = PredAugSigmaPoints.col(i)[1];
//     double vx = PredAugSigmaPoints.col(i)[2]*cos(PredAugSigmaPoints.col(i)[3]);
//     double vy = PredAugSigmaPoints.col(i)[2]*sin(PredAugSigmaPoints.col(i)[3]);
//     double rho = sqrt(px*px + py*py);
//     double eps = 0.00001;
//     double phi;
//     if(py >= 0){
//         phi = (fabs(px) >= eps) ? atan2(py, px) : M_PI/2.0;
//     }
//     else{
//         phi = (fabs(px) >= eps) ? atan2(py, px) : -M_PI/2.0;
//     }
//     double rho_dot = (px*vx+py*vy)/std::max(eps, rho);
//     RadarSigmaPoints.col(i) << rho, phi, rho_dot;
//   }
//   // cout << "RadarSigma: " << RadarSigmaPoints << endl;
//   VectorXd z_pred(n_z); //the mean vector of radar prediction measurement
//   MatrixXd S(n_z, n_z); // the covariance matrix of radar prediciton measurement
//   //fill z_pred
//   z_pred.fill(0.0);
//   for(int i = 0; i < 2*n_aug_+1; i++)
//     z_pred += weights_[i]*RadarSigmaPoints.col(i);
//   //fill S
//   S.fill(0.0);
//   for(int i = 0; i < 2*n_aug_+1; i++){
//     VectorXd diff = RadarSigmaPoints.col(i) - z_pred;
//     while(diff(1) > M_PI) diff(1) -= 2*M_PI;
//     while(diff(1) < -M_PI) diff(1) += 2*M_PI;
//     S += weights_(i)*diff*diff.transpose();
//   }
//   S += R_Radar_;
//   PredRadar = VectorXd(n_z);
//   PredRadar = z_pred;
//   PredRadarCovariance = MatrixXd(n_z, n_z);
//   PredRadarCovariance = S;
// }

// void UKF::RadarUpdate(MeasurementPackage meas_package)
// {
//   //Predict radar measurement vector and covariance matrix
//   PredictRadarMeasurement();
//   int n_z = 3;
//   MatrixXd T(n_x_, n_z);
//   T.fill(0.0);
//   for(int i = 0; i < 2 * n_aug_+1; i++){
//     VectorXd diff_x = PredAugSigmaPoints.col(i) - x_;
//     while(diff_x(3) > M_PI) diff_x(3) -= 2*M_PI;
//     while(diff_x(3) < -M_PI) diff_x(3) += 2*M_PI;
//     VectorXd diff_z = RadarSigmaPoints.col(i) - PredRadar;
//     while(diff_z(1) > M_PI) diff_z(1) -= 2*M_PI;
//     while(diff_z(1) < -M_PI) diff_z(1) += 2*M_PI;
//     T += weights_(i) * diff_x * diff_z.transpose();
//   }
//   //Kalman gain
//   MatrixXd K = T*PredRadarCovariance.inverse();
//   //update state mean vector
//   x_ = x_ + K*(meas_package.raw_measurements_ - PredRadar);
//   //update state covariance
//   P_ -= K*PredRadarCovariance*K.transpose();
//   NIS_Radar_ = (meas_package.raw_measurements_ - PredRadar).transpose() * PredRadarCovariance.inverse() *
//         (meas_package.raw_measurements_ - PredRadar);
// }

#endif /* UKF_H */
