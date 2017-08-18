#include "PID.h"

using namespace std;

PID::PID() {
    cte_ = 0.0;
    cte_sum_ = 0.0;
    cte_diff_ = 0.0;
    cte_pre_ = 0.0;
}

PID::~PID() {}

void PID::Init(double Kp, double Ki, double Kd) {
    Kp_ = Kp;
    Ki_ = Ki;
    Kd_ = Kd;
}

void PID::UpdateError(double cte) {
    cte_ = cte;
    cte_sum_ += cte;
    cte_diff_ = cte - cte_pre_;
    cte_pre_ = cte;
}

double PID::TotalError() {
    double total_error = -(Kp_*cte_ + Ki_*cte_sum_ + Kd_*cte_diff_);
    return total_error;
}

