#ifndef WAYPOINTS_H_
#define WAYPOINTS_H_

#include "helper.h"
using namespace std;

//find the index of closest point in maps of highway double yellow line points to (x, y)
int ClosestWaypoint(double x, double y, vector<double> maps_x, vector<double> maps_y)
{

    double closestLen = 100000; //large number
    int closestWaypoint = 0;

    for(int i = 0; i < maps_x.size(); i++)
    {
        double map_x = maps_x[i];
        double map_y = maps_y[i];
        double dist = distance(x,y,map_x,map_y);
        if(dist < closestLen)
        {
            closestLen = dist;
            closestWaypoint = i;
        }

    }

    return closestWaypoint;

}
//if the difference between angle of nearest double yellow waypoint with the point where you are and the car's heading is more than 45 degree,
//we believe that the closest waypoint is behind the direction we want to move forward
//thus we add the index of the nearest waypoint
int NextWaypoint(double x, double y, double theta, vector<double> maps_x, vector<double> maps_y)
{

    int closestWaypoint = ClosestWaypoint(x,y,maps_x,maps_y);

    double map_x = maps_x[closestWaypoint];
    double map_y = maps_y[closestWaypoint];

    double heading = atan2( (map_y-y),(map_x-x) );

    double angle = abs(theta-heading);

    if(angle > pi()/4)
    {
        closestWaypoint++;
    }

    return closestWaypoint;

}

#endif