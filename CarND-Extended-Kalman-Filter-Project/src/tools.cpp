#include "tools.h"
#include <iostream>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
   * TODO: Calculate the RMSE here.
   */

    VectorXd rmse(4);
    rmse << 0,0,0,0;

    // some validation
    if(estimations.size() == 0){
      std::cout << " [ Error ] `estimations` is empty" << std::endl;
      return rmse;
    }

    if(ground_truth.size() == 0){
      std::cout << " [ Error ] `ground-truth` is empty" << std::endl;
      return rmse;
    }

    if(estimations.size() != ground_truth.size()){
      std::cout << " [ Error ] Shape mismatch" << std::endl;
      return rmse;
    }

    for(unsigned int i=0; i < estimations.size(); ++i){
      VectorXd resi = estimations[i] - ground_truth[i];
      resi = resi.array() * resi.array();
      rmse += resi;
    }
    unsigned int l = estimations.size();
    rmse = rmse / l;
    rmse = rmse.array().sqrt();
    return rmse;

}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  /**
   * TODO:
   * Calculate a Jacobian here.
   */

  MatrixXd Hj(3,4);

  if ( x_state.size() != 4 ) {
    std::cout << " [Error] Invalid Shape!" << std::endl;
    return Hj;
  }

  double px = x_state(0);
  double py = x_state(1);
  double vx = x_state(2);
  double vy = x_state(3);

  double c1 = px*px+py*py;
  double c2 = sqrt(c1);
  double c3 = (c1*c2);

  //check for division by zero
  if (fabs(c1) < 0.0001){
      std::cout << "Error - Division by Zero" << std::endl;
      return Hj;
  }

  //compute the Jacobian matrix
  Hj << (px / c2), (py / c2), 0, 0,
      -(py / c1), (px / c1), 0, 0,
      py*(vx*py - vy*px) / c3, px*(px*vy - py*vx) / c3, px / c2, py / c2;

  return Hj;


}

