#include "kalman_filter.h"

#include <math.h>
using Eigen::MatrixXd;
using Eigen::VectorXd;

/* 
 * Please note that the Eigen library does not initialize 
 *   VectorXd or MatrixXd objects with zeros upon creation.
 */

KalmanFilter::KalmanFilter() {

  I = MatrixXd::Identity(4, 4);
}

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
  x_ = F_ * x_;					// prior
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;		// state variance
}

void KalmanFilter::Update(const VectorXd &z) {
VectorXd y = z - H_ * x_;		// residual
MatrixXd Ht = H_.transpose();
MatrixXd S = H_ * P_ * Ht + R_;	// system uncertainty
MatrixXd Si = S.inverse();
MatrixXd K =  P_ * Ht * Si;  	// Kalman gain
// new state
x_ = x_ + (K * y);				// posterior
P_ = (I - K * H_) * P_;			// posterior variance  
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
float px = x_(0);
float py = x_(1);
float vx = x_(2);
float vy = x_(3);
  
float rho = sqrt(px*px + py*py);
float phi = atan2(py, px);
float rho_dot = (px*vx + py*vy)/rho;
  
VectorXd z_pred(3);
z_pred << rho, phi, rho_dot;  
VectorXd y = z - z_pred;	// residual
  
// normalize phi between -pi and pi
if (y[1] > M_PI) {
  y[1] = fmod(y[1], M_PI);
} else if (y[1] < M_PI * -1) {
  y[1] = (M_PI * -1) + fmod(y[1], M_PI);
}

MatrixXd Ht = H_.transpose();
MatrixXd S = H_ * P_ * Ht + R_;	// system uncertainty
MatrixXd Si = S.inverse();
MatrixXd PHt = P_ * Ht;
MatrixXd K = PHt * Si;	// Kalman gain
x_ = x_ + (K * y);		// posterior
P_ = (I - K * H_) * P_;	// posterior variance
}
