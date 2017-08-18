## Reflection
####1. The effects of P, I and D component in the implementation.
For P component, it makes the car oscillate around the planned trajectory. And It guarantees that the car will not get far away from the trajectory.

For I component, it will reduce the influence of **drift** when the wheels are not 100% aligned.

For D component, it will diminish the effect of oscillation and make the actual trajectory more smooth that just using P control.

####2. How to choose the PID controller coefficients.
I just use manual tuning. First I use relative large parameters $Kp = 1.0, Ki = 0.2, Kd = 10.0$. The car will drive away from the road. The I gradually tune down the parameters, and finally I choose $Kp = 0.1, Ki = 0.001, Kd = 1.0$

