#include <iostream>
#include "PID.h"

using namespace std;

int main()
{
    PID pid;
    cout << "cte: " << pid.cte_ << endl;
    pid.Init(0.2, 0.5, 3.0);
    cout << "Kp: " << pid.Kp_ << "Ki: " << pid.Ki_ << "Kd: " << pid.Kd_ << endl;
    pid.UpdateError(2.0);
    cout << "cte: " << pid.cte_ << "cte_sum: " << pid.cte_sum_ << "cte_diff: " << pid.cte_diff_ << endl;
    return 0;
}