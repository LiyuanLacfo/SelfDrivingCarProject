# Reflection
This is the reflection on Udacity Self-driving car nanodegree Projection 10 **Model Predictive Control**.

### Model
The MPC includes 6 state variables which are x coordinate *x*, y coordinate *y*, the orientation of vehicle *$\psi$*, the velocity *v*, the cross track error *cte*, and the difference between planned path orientation and the vehicle orientation *$e\psi$*.

The actuators contains the steering angle *$\delta$* and brake/throttle $a$.

The update model,

  $x_{t+1} = x_t + v_t*cos(\psi_t)*dt$
            $y_{t+1} = y_t + v_t*sin(\psi_t)*dt$
            $\psi_{t+1} = \psi_t + v_t*\delta_t*dt/Lf$
            $v_{t+1} = v_t + a_t * dt$
            $cte_{t+1} = f(x_t) - y_t + v_t*sin(e\psi_t)*dt$
            $e\psi_{t+1} = \psi_t - \psi_{desired} + v_t*\delta_t * dt/Lf$
            
### Timestep Length and Elapsed Duration (N & dt)
In my project, $N=25, dt = 0.04$.

$T = N * dt$. And $T$ can't be too large, since the control values derived from MPC are just approximate. If $T$ is large such as 10 seconds, the error would be large.

Also $dt$ can't be large, otherwise the path predicted would be less smooth.

I have tried $N = 100, dt = 0.04$, the cross track error would be larger than the values above.

### Polynomial Fitting and MPC Preprocessing
The waypoints we derive from the simulator are in the global coordinate. Thus if we want to fit these points, we need first transform these points into the vehicle coordinate.

After transformation, the vehicle would be at the origin. Then it is easy to calculate *cte* by $f(0) - 0$. Because the orientation of vehicle is now the x axis and vehicle is at $(0, 0)$, $e\psi = atan(coeffs[1])$.

### Model Predictive Control with Latency
To address latency, I just use the model above to predict the state vector after duration of latency. And then I use this predictive state vector to feed into the solver. 
            



