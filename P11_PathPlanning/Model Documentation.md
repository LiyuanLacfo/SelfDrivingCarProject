## Reflection
This is the reflection of the *path planning* project. Here I would describe the details of generating the path.

For the path planning on the highway, we need to decide whether to stay on the current lane or to change to the next lanes. 

Every time for the decision making, we will first go through the **sensor fusion** data. It can tell us the information on the velocity and position of the vehicles around us. 

First if there is a car in the same lane as our car and very close to use, we need to slow down. Then we should consider whether it is appropriate to change lanes.

Next if we need to see whether the next lanes are empty, we should consider the distance of the cars in the next lanes if they exists. If the there is a car that is very close to us, then it is not safe to change lanes. If the distance is large enough that we can safely change the lanes, then we should do that to pass the slow car in front of us.

The final question is how to generate path points that we are going to reach if we want to keep the lane or change lanes. Cubic spline interpolating is a good idea. First we need to generate some points to fit under the Frenet coordinate. Then we transform these points into Cartesian coordinate. To make the problem more easier, we can transform these points from global Cartesian coordinate to the local Cartesian coordinate with the reference autonomous car position as the origin. Then we fit the cubic spline within the local coordinate.

To keep the acceleration and jerk within the corresponding limits, we need to use the reference velocity to decide how many points we want to generate within some range. After that number is decided, we can generate points. But these points are in the local coordinate, we must transform them into global coordinate. 

That is the process of generating path.

