#ifndef HELPER_H_
#define HELPER_H_

#include <cmath>
#include "json.hpp"
#include <vector>
using namespace std;

struct Vehicle
{
    double x, y;
    double d;
    double s;
    double vx, vy;
    double speed;
    double heading;
    Vehicle(vector<double> vec)
    {
        this -> x = vec[1]; this -> y = vec[2];
        this -> vx = vec[3]; this -> vy = vec[4];
        this -> s = vec[5]; this -> d = vec[6];
        this -> speed = sqrt(vx*vx + vy*vy);
        this -> heading = atan2(vy, vx);
    }
};

const double speed_limit = 49.5; //mph
const double lane_width = 4.0;
const double safety_margin = 30.0;
const int n_points = 50;
const double frame_sec = 0.02;
const double angle_gap = M_PI/4;

constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }
double distance(double x1, double y1, double x2, double y2)
{
    return sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1));
}

bool is_in_lane(double d, int lane)
{
    return (d > (lane_width*lane) && d < (lane_width + lane_width*lane));
}

bool dangerous_to_change_lane(double ego_s, double vehicle_s, double ego_speed, double vehicle_speed)
{
    return (((ego_s > vehicle_s - safety_margin/2) && (ego_s < vehicle_s + safety_margin/2)) 
             || ((vehicle_s < ego_s - safety_margin/2) && (vehicle_s + vehicle_speed > ego_s))
             || ((vehicle_s > ego_s + safety_margin/2) && (ego_s + ego_speed/2.4 > vehicle_s)));
}

#endif