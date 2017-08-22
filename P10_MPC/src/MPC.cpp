#include "MPC.h"
#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve.hpp>
#include "Eigen-3.3/Eigen/Core"
#include <vector>

using CppAD::AD;

//Set the timestep length and duration
size_t N = 25;
double dt = 0.04;

//set the start index of each state variable in var list for optimization 
size_t x_start      = 0;
size_t y_start      = x_start     +   N;
size_t psi_start    = y_start     +   N;
size_t v_start      = psi_start   +   N;
size_t cte_start    = v_start     +   N;
size_t epsi_start   = cte_start   +   N;
size_t delta_start  = epsi_start  +   N;
size_t a_start      = delta_start +   N - 1;

//set the reference velocity
double ref_v = 40.0;


// It was obtained by measuring the radius formed by running the vehicle in the
// simulator around in a circle with a constant steering angle and velocity on a
// flat terrain

// This is the length from front to CoG that has a similar radius.
const double Lf = 2.67;

class FG_eval {
 public:
  // Fitted polynomial coefficients
  Eigen::VectorXd coeffs;
  FG_eval(Eigen::VectorXd coeffs) { this->coeffs = coeffs; }

  typedef CPPAD_TESTVECTOR(AD<double>) ADvector;
  void operator()(ADvector& fg, const ADvector& vars) {
    //set the cost function
    fg[0] = 0.0;
    //define the weights of cost for each component
    const double cte_weight = 2000;
    const double epsi_weight = 2000;
    const double v_weight = 100;
    const double actuator_cost_weight = 1000;
    const double change_steer_cost_weight = 1000000;
    const double change_accel_cost_weight = 10000;
    //add cte and epsi and velocity
    for(int i = 0; i < N; i++){
      fg[0] += cte_weight * CppAD::pow(vars[cte_start + i], 2);//add cte
      fg[0] += epsi_weight * CppAD::pow(vars[epsi_start + i], 2); //add epsi
      fg[0] += v_weight * CppAD::pow(vars[v_start + i] - ref_v, 2); // add velocity
    }
    //add actuator
    for(int i = 0; i < N-1; i++){
      fg[0] += actuator_cost_weight * CppAD::pow(vars[delta_start + i], 2);
      fg[0] += actuator_cost_weight * CppAD::pow(vars[a_start + i], 2);
    }
    //add smooth of actuator
    for(int i = 0; i < N-2; i++){
      fg[0] += change_steer_cost_weight * CppAD::pow(vars[delta_start + i + 1]-vars[delta_start + i], 2);
      fg[0] += change_accel_cost_weight * CppAD::pow(vars[a_start + i + 1]-vars[a_start + i], 2);
    }

    //Set the constraints
    //set the initial state constraint
    fg[1+x_start] = vars[x_start];
    fg[1+y_start] = vars[y_start];
    fg[1+v_start] = vars[v_start];
    fg[1+psi_start] = vars[psi_start];
    fg[1+cte_start] = vars[cte_start];
    fg[1+epsi_start] = vars[epsi_start];
    //set other constraints
    for(int i = 1; i < N; i++){
      //current
      AD<double> x0 = vars[x_start+i-1];
      AD<double> y0 = vars[y_start+i-1];
      AD<double> v0 = vars[v_start+i-1];
      AD<double> psi0 = vars[psi_start+i-1];
      AD<double> cte0 = vars[cte_start+i-1];
      AD<double> epsi0 = vars[epsi_start+i-1];

      //next
      AD<double> x1 = vars[x_start+i];
      AD<double> y1 = vars[y_start+i];
      AD<double> v1 = vars[v_start+i];
      AD<double> psi1 = vars[psi_start+i];
      AD<double> cte1 = vars[cte_start+i];
      AD<double> epsi1 = vars[epsi_start+i];

      //a0
      AD<double> a0 = vars[a_start+i-1];
      //delta0
      AD<double> delta0 = vars[delta_start+i-1];

      //f0
      AD<double> f0 = coeffs[0] + coeffs[1]*x0 + coeffs[2]*x0*x0 + coeffs[3]*x0*x0*x0;
      //psides0:desired psi
      AD<double> psides0 = CppAD::atan(coeffs[1] + 2*coeffs[2]*x0 + 3*coeffs[3]*x0*x0);

      //set up constraints
      fg[x_start+1+i] = x1 - x0 - v0*CppAD::cos(psi0)*dt;
      fg[y_start+1+i] = y1 - y0 - v0*CppAD::sin(psi0)*dt;
      fg[psi_start+1+i] = psi1 - psi0 + v0*delta0*dt/Lf;
      fg[v_start+1+i] = v1 - v0 - a0*dt;
      fg[cte_start+1+i] = cte1 - (f0 - y0) - v0*CppAD::sin(epsi0)*dt;
      fg[epsi_start+1+i] = epsi1 - (psi0 - psides0) + v0*delta0*dt/Lf;
    }

  }
};

//
// MPC class definition implementation.
//
MPC::MPC() {}
MPC::~MPC() {}

vector<double> MPC::Solve(Eigen::VectorXd state, Eigen::VectorXd coeffs) {
  bool ok = true;
  // size_t i;
  typedef CPPAD_TESTVECTOR(double) Dvector;
  //extract initial state
  double x = state[0];
  double y = state[1];
  double psi = state[2];
  double v = state[3];
  double cte = state[4];
  double epsi = state[5];

  // set the number of variables
  size_t n_vars = 6*N + 2*(N-1);
  // Set the number of constraints
  size_t n_constraints = 6*N;

  // Initial value of the independent variables.
  // SHOULD BE 0 besides initial state.
  Dvector vars(n_vars);
  for (int i = 0; i < n_vars; i++) {
    vars[i] = 0;
  }
  vars[x_start] = x;
  vars[y_start] = y;
  vars[psi_start] = psi;
  vars[v_start] = v;
  vars[cte_start] = cte;
  vars[epsi_start] = epsi;

  Dvector vars_lowerbound(n_vars);
  Dvector vars_upperbound(n_vars);
  //Set lower and upper limits for variables.
  for(int i = 0; i < delta_start; i ++){
    vars_lowerbound[i] = -1.0e19;
    vars_upperbound[i] = 1.0e19;
  }
  //set the lower and upper bound for delta, between -25 degree to 25 degree
  for(int i = delta_start; i < a_start; i++){
    vars_lowerbound[i] = -0.436332;
    vars_upperbound[i] = 0.436332;
  }
  //set the lower bound and upper bound for a, between -1 and 1
  for(int i = a_start; i < n_vars; i++){
    vars_lowerbound[i] = -1;
    vars_upperbound[i] = 1;
  }

  // Lower and upper limits for the constraints
  // Should be 0 besides initial state.
  Dvector constraints_lowerbound(n_constraints);
  Dvector constraints_upperbound(n_constraints);
  for (int i = 0; i < n_constraints; i++) {
    constraints_lowerbound[i] = 0;
    constraints_upperbound[i] = 0;
  }
  constraints_lowerbound[x_start] = x;
  constraints_upperbound[x_start] = x;

  constraints_lowerbound[y_start] = y;
  constraints_upperbound[y_start] = y;

  constraints_lowerbound[psi_start] = psi;
  constraints_upperbound[psi_start] = psi;

  constraints_lowerbound[v_start] = v;
  constraints_upperbound[v_start] = v;

  constraints_lowerbound[cte_start] = cte;
  constraints_upperbound[cte_start] = cte;

  constraints_lowerbound[epsi_start] = epsi;
  constraints_upperbound[epsi_start] = epsi;

  // object that computes objective and constraints
  FG_eval fg_eval(coeffs);

  //
  // options for IPOPT solver
  std::string options;
  // Uncomment this if you'd like more print information
  options += "Integer print_level  0\n";
  // NOTE: Setting sparse to true allows the solver to take advantage
  // of sparse routines, this makes the computation MUCH FASTER. If you
  // can uncomment 1 of these and see if it makes a difference or not but
  // if you uncomment both the computation time should go up in orders of
  // magnitude.
  options += "Sparse  true        forward\n";
  options += "Sparse  true        reverse\n";
  // NOTE: Currently the solver has a maximum time limit of 0.5 seconds.
  // Change this as you see fit.
  options += "Numeric max_cpu_time          0.5\n";

  // place to return solution
  CppAD::ipopt::solve_result<Dvector> solution;

  // solve the problem
  CppAD::ipopt::solve<Dvector, FG_eval>(
      options, vars, vars_lowerbound, vars_upperbound, constraints_lowerbound,
      constraints_upperbound, fg_eval, solution);

  // Check some of the solution values
  ok &= solution.status == CppAD::ipopt::solve_result<Dvector>::success;

  // Cost
  auto cost = solution.obj_value;
  std::cout << "Cost " << cost << std::endl;
  
  std::vector<double> result;
  auto a0 = solution.x[a_start]; 
  auto delta0 = solution.x[delta_start];
  result.push_back(delta0);
  result.push_back(a0);
  for(int i = 0; i < N; ++i){
    result.push_back(solution.x[x_start+i]);
    result.push_back(solution.x[y_start+i]);
  }
  return result;
}
