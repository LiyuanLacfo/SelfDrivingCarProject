#include "kalman_filter.h"

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
  TODO:
    * predict the state
  */
  x_ = F_*x_;
  MatrixXd Ft;
  Ft = F_.transpose();
  P_ = F_*P_*Ft + Q_;
}

void KalmanFilter::UpdateCommon(const VectorXd &y){
  MatrixXd Ht = H_.transpose();
  MatrixXd S;
  S = H_*P_*Ht + R_;
  MatrixXd Sinv = S.inverse();
  MatrixXd K = P_*Ht*Sinv;
  x_ = x_ + K*y;
  MatrixXd I = MatrixXd::Identity(x_.size(), x_.size());
  P_ = (I-K*H_)*P_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Kalman Filter equations
  */
  VectorXd y(2);
  y = z - H_*x_;
  cout << "y: " << y << endl;
  UpdateCommon(y);

}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Extended Kalman Filter equations
  */
  float px = x_(0);
  float py = x_(1);
  float vx = x_(2);
  float vy = x_(3);
  VectorXd y(3);
  VectorXd z_pred(3);
  double rho = sqrt(px*px + py*py);
  double phi = atan2(py, px);
  double rho_dot = (px*vx + py*vy)/std::max(0.0001, rho);
  z_pred << rho, phi, rho_dot;
  y = z - z_pred;
  //normalize y
  while(y(1) > M_PI) y(1) -= 2*M_PI;
  while(y(1) < -M_PI) y(1) += 2*M_PI;
  UpdateCommon(y);
}

void KalmanFilter::UpdateF(const double dt){
  F_ = MatrixXd(4, 4);
  F_ << 1, 0, dt, 0,
        0, 1, 0, dt,
        0, 0, 1, 0,
        0, 0, 0, 1;
}

void KalmanFilter::UpdateQ(const double dt){
  /**
   TODO:
     * Update the state transition matrix F according to the new elapsed time.
      - Time is measured in seconds.
     * Update the process noise covariance matrix.
     * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */
  MatrixXd G(4, 2);
  G << dt*dt/2, 0,
       0, dt*dt/2,
       dt, 0,
       0, dt;
  MatrixXd Gt = G.transpose();
  MatrixXd T(2, 2);
  T << 9, 0,
       0, 9;
  Q_ = G*T*Gt;
}

