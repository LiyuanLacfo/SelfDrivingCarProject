#ifndef PID_H
#define PID_H

class PID {
public:
  /*
  * Errors
  */
  double cte_; //current cte
  double cte_sum_; // the sum of cte
  double cte_diff_; // the difference between current cte and previous cte

  //the previous cte
  double cte_pre_;

  /*
  * Coefficients
  */ 
  double Kp_;
  double Ki_;
  double Kd_;

  /*
  * Constructor
  */
  PID();

  /*
  * Destructor.
  */
  virtual ~PID();

  /*
  * Initialize PID.
  */
  void Init(double Kp, double Ki, double Kd);

  /*
  * Update the PID error variables given cross track error.
  */
  void UpdateError(double cte);

  /*
  * Calculate the total PID error.
  */
  double TotalError();
};


#endif /* PID_H */
