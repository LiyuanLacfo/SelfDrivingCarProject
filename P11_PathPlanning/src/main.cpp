#include <fstream>
#include <algorithm>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "spline.h"
#include "coordinate_transform.h"

using namespace std;

// for convenience
using json = nlohmann::json;

bool cmp(const vector<double> &a, const vector<double> &b)
{
  return a[0] < b[0];
}

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.find_first_of("}");
  if (found_null != string::npos) {
    return "";
  } else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 2);
  }
  return "";
}



int main() {
  uWS::Hub h;

  // Load up map values for waypoint's x,y,s and d normalized normal vectors
  vector<double> map_waypoints_x;
  vector<double> map_waypoints_y;
  vector<double> map_waypoints_s;
  vector<double> map_waypoints_dx;
  vector<double> map_waypoints_dy;

  // Waypoint map to read from
  string map_file_ = "../data/highway_map.csv";
  // The max s value before wrapping around the track back to 0
  double max_s = 6945.554;

  ifstream in_map_(map_file_.c_str(), ifstream::in);

  string line;
  while (getline(in_map_, line)) {
    istringstream iss(line);
    double x;
    double y;
    float s;
    float d_x;
    float d_y;
    iss >> x;
    iss >> y;
    iss >> s;
    iss >> d_x;
    iss >> d_y;
    map_waypoints_x.push_back(x);
    map_waypoints_y.push_back(y);
    map_waypoints_s.push_back(s);
    map_waypoints_dx.push_back(d_x);
    map_waypoints_dy.push_back(d_y);
  }
   //the starting lane
  int lane = 1;
  //the reference velocity(miles per hour)
  double ref_v = 0.0;

  h.onMessage([&ref_v, &map_waypoints_x,&map_waypoints_y,&map_waypoints_s,&map_waypoints_dx,&map_waypoints_dy, &lane](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                     uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    //auto sdata = string(data).substr(0, length);
    //cout << sdata << endl;
    if (length && length > 2 && data[0] == '4' && data[1] == '2') {

      auto s = hasData(data);

      if (s != "") {
        auto j = json::parse(s);
        
        string event = j[0].get<string>();
        
        if (event == "telemetry") {
          // j[1] is the data JSON object
          
            // Main car's localization Data
            double car_x = j[1]["x"];
            double car_y = j[1]["y"];
            double car_s = j[1]["s"];
            double car_d = j[1]["d"];
            double car_yaw = j[1]["yaw"];
            double car_speed = j[1]["speed"];

            // Previous path data given to the Planner
            auto previous_path_x = j[1]["previous_path_x"];
            auto previous_path_y = j[1]["previous_path_y"];
            // Previous path's end s and d values 
            double end_path_s = j[1]["end_path_s"];
            double end_path_d = j[1]["end_path_d"];

            // Sensor Fusion Data, a list of all other cars on the same side of the road.
            vector<vector<double>> sensor_fusion = j[1]["sensor_fusion"];

            json msgJson;

            //if previous points almost nothing
            int prev_size = previous_path_x.size();
            //if prev_size >0, use end_path_s to replace car_s
            if(prev_size > 0)
              car_s = end_path_s;
            //define the finite state machine
            bool too_close = false;
            bool prepare_lane_change = false;
            bool left_empty = true;
            bool right_empty = true;
            bool watch_out_next_lane = false;

            //find whether ego car is very close to the car in front of it in the same lane
            for(size_t i = 0; i < sensor_fusion.size(); i++)
            {
              Vehicle vehicle(sensor_fusion[i]);
              double d = vehicle.d;
              cout << "vx: " << vehicle.vx << " vy: " << vehicle.vy << endl;
              vehicle.s += (double)prev_size*frame_sec*vehicle.speed; //we are looking where the car is in the future
              if(is_in_lane(d, lane))
              {
                bool is_in_front_of_us = vehicle.s > car_s;
                bool is_closer_than_safety_margin = vehicle.s - car_s < safety_margin;
                if(is_in_front_of_us && is_closer_than_safety_margin)
                {
                  too_close = true;
                  prepare_lane_change = true;
                }
              }
              else if(is_in_lane(d, lane-1))
              {
                bool is_in_front_of_us = vehicle.s > car_s;
                bool is_closer_than_safety_margin = vehicle.s - car_s < safety_margin;
                bool next_lane_too_close = false;
                next_lane_too_close = is_in_front_of_us && is_closer_than_safety_margin;
                int idx = NextWaypoint(vehicle.x, vehicle.y, vehicle.heading, map_waypoints_x, map_waypoints_y);
                int pre_idx = (idx-1) % map_waypoints_x.size();
                double cur_x = map_waypoints_x[idx];
                double pre_x = map_waypoints_x[pre_idx];
                double cur_y = map_waypoints_y[idx];
                double pre_y = map_waypoints_y[pre_idx];
                double way_heading = atan2(cur_y-pre_y, cur_x-pre_x);
                if(vehicle.heading < 0.0)
                  vehicle.heading += 2*pi();
                if(way_heading<0.0)
                  way_heading += 2*pi();
                double change_lane_to_me = false;
                if(way_heading-vehicle.heading>angle_gap)
                  change_lane_to_me = true;
                if(next_lane_too_close && change_lane_to_me)
                  watch_out_next_lane = true;
              }
              else if(is_in_lane(d, lane+1))
              {
                bool is_in_front_of_us = vehicle.s > car_s;
                bool is_closer_than_safety_margin = vehicle.s - car_s < safety_margin;
                bool next_lane_too_close = false;
                next_lane_too_close = is_in_front_of_us && is_closer_than_safety_margin;
                int idx = NextWaypoint(vehicle.x, vehicle.y, vehicle.heading, map_waypoints_x, map_waypoints_y);
                int pre_idx = (idx-1) % map_waypoints_x.size();
                double cur_x = map_waypoints_x[idx];
                double pre_x = map_waypoints_x[pre_idx];
                double cur_y = map_waypoints_y[idx];
                double pre_y = map_waypoints_y[pre_idx];
                double way_heading = atan2(cur_y-pre_y, cur_x-pre_x);
                if(vehicle.heading < 0.0)
                  vehicle.heading += 2*pi();
                if(way_heading < 0.0)
                  way_heading += 2*pi();
                double change_lane_to_me = false;
                if(vehicle.heading - way_heading > angle_gap)
                  change_lane_to_me = true;
                if(next_lane_too_close && change_lane_to_me)
                  watch_out_next_lane = true;
              }
            }
            //if prepare_lane_change is true, we want to decide whether the left lane or right lane is emtpy
            double nearest_left_delta_s = 100000.0;
            double nearest_right_delta_s = 100000.0;
            if(prepare_lane_change)
            {
              for(size_t i = 0; i < sensor_fusion.size(); i++)
              {
                Vehicle vehicle(sensor_fusion[i]);
                vehicle.s += (double)prev_size*frame_sec*vehicle.speed;
                //whether left lane or right lane has car
                if(is_in_lane(vehicle.d, lane-1))
                {
                  if(vehicle.s > car_s)
                  {
                    double delta_s = vehicle.s - car_s;
                    if(delta_s < nearest_left_delta_s)
                      nearest_left_delta_s = delta_s;
                  }
                  bool is_too_close_to_change = dangerous_to_change_lane(car_s, vehicle.s, car_speed, vehicle.speed);
                  if(is_too_close_to_change)
                    left_empty = false;
                }
                else if(is_in_lane(vehicle.d, lane+1))
                {
                  if(vehicle.s > car_s)
                  {
                    double delta_s = vehicle.s - car_s;
                    if(delta_s < nearest_right_delta_s)
                      nearest_right_delta_s = delta_s;
                  }
                  bool is_too_close_to_change = dangerous_to_change_lane(car_s, vehicle.s, car_speed, vehicle.speed);
                  if(is_too_close_to_change)
                    right_empty = false;
                }
              }
            }
            if(too_close)
              ref_v -= 0.224;
            else if(ref_v < speed_limit)
              ref_v += 0.224;
            if(watch_out_next_lane)
              ref_v -= 0.224;
            bool able_to_left = prepare_lane_change && left_empty && (lane > 0);
            bool able_to_right = prepare_lane_change && right_empty && (lane < 2);
            if(able_to_left && !able_to_right)
              lane -= 1;
            else if(!able_to_left && able_to_right)
              lane += 1;
            else if(able_to_left && able_to_right)
            {
              if(nearest_left_delta_s > nearest_right_delta_s)
                lane -= 1;
              else
                lane += 1;
            }
            //generate spline for trajectory
            //create a vector of x and y points that will be used later for interpolating
            vector<double> ptsx;
            vector<double> ptsy;
            //reference x, y and yaw for later coordinate transformation
            double ref_x = car_x;
            double ref_y = car_y;
            double ref_yaw = deg2rad(car_yaw);
            if(prev_size < 2)
            {
              //generate a point that the line between it and (car_x, car_y) is tangent to the path
              double prev_car_x = car_x - cos(ref_yaw);
              double prev_car_y = car_y - sin(ref_yaw);
              ptsx.push_back(prev_car_x);
              ptsx.push_back(car_x);
              ptsy.push_back(prev_car_y);
              ptsy.push_back(car_y);
            }
            else
            {
              //redefine the reference x and y
              ref_x = previous_path_x[prev_size-1];
              ref_y = previous_path_y[prev_size-1];
              double prev_ref_x = previous_path_x[prev_size-2];
              double prev_ref_y = previous_path_y[prev_size-2];
              ref_yaw = atan2((ref_y - prev_ref_y), (ref_x - prev_ref_x));

              ptsx.push_back(prev_ref_x);
              ptsx.push_back(ref_x);
              ptsy.push_back(prev_ref_y);
              ptsy.push_back(ref_y);
            }
            //generate 3 points spaced 30 meters from the reference (x, y)
            vector<double> next_wp0 = getXY(car_s + 30, (lane_width/2+lane_width*lane), map_waypoints_s, map_waypoints_x, map_waypoints_y);
            vector<double> next_wp1 = getXY(car_s + 60, (lane_width/2+lane_width*lane), map_waypoints_s, map_waypoints_x, map_waypoints_y);
            vector<double> next_wp2 = getXY(car_s + 90, (lane_width/2+lane_width*lane), map_waypoints_s, map_waypoints_x, map_waypoints_y);

            ptsx.push_back(next_wp0[0]);
            ptsx.push_back(next_wp1[0]);
            ptsx.push_back(next_wp2[0]);

            ptsy.push_back(next_wp0[1]);
            ptsy.push_back(next_wp1[1]);
            ptsy.push_back(next_wp2[1]);

            //transformation of the coordinate system with the origin at (ref_x, ref_y)
            for(int i = 0; i < ptsx.size(); i++)
            {
              double shift_x = ptsx[i] - ref_x;
              double shift_y = ptsy[i] - ref_y;
              ptsx[i] = shift_x*cos(ref_yaw) + shift_y*sin(ref_yaw);
              ptsy[i] = -shift_x*sin(ref_yaw) + shift_y*cos(ref_yaw);
            }
            vector<vector<double>> combined;
            for(int i = 0; i < ptsx.size(); i++)
            {
              combined.push_back({ptsx[i], ptsy[i]});
            }
            sort(combined.begin(), combined.end(), cmp);
            for(int i = 0; i < ptsx.size(); i++)
            {
              ptsx[i] = combined[i][0];
              ptsy[i] = combined[i][1];
            }
            //create a spline
            tk::spline s;
            //set x and y points to spline
            s.set_points(ptsx, ptsy);
            //start to fill next x and y
            vector<double> next_x_vals;
            vector<double> next_y_vals;
            for(int i = 0; i < previous_path_x.size(); i++)
            {
              next_x_vals.push_back(previous_path_x[i]);
              next_y_vals.push_back(previous_path_y[i]);
            }
            //calculate how to break up spline points so that we can meet our desired speed 
            double target_x = 30.0;
            double target_y = s(target_x);
            double target_dist = distance(0.0, 0.0, target_x, target_y);//here we use the local coordinate

            double x_add_on = 0.0;
            double N = target_dist/(ref_v/2.24*frame_sec);//calculate the desired N for interpolating, meters per second
            for(int i = prev_size; i < n_points; i++)
            {
              double x_point;
              x_point = x_add_on + target_x / N;
              double y_point = s(x_point);
              x_add_on = x_point;
              
              double x_ref = x_point;
              double y_ref = y_point;
              //rotate back to the global coordinate
              x_point = x_ref*cos(ref_yaw) - y_ref*sin(ref_yaw);
              y_point = x_ref*sin(ref_yaw) + y_ref*cos(ref_yaw);

              x_point += ref_x;
              y_point += ref_y;
              next_x_vals.push_back(x_point);
              next_y_vals.push_back(y_point);
            }
            msgJson["next_x"] = next_x_vals;
            msgJson["next_y"] = next_y_vals;

            auto msg = "42[\"control\","+ msgJson.dump()+"]";

            //this_thread::sleep_for(chrono::milliseconds(1000));
            ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
          
        }
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }
  });

  // We don't need this since we're not using HTTP but if it's removed the
  // program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
                     size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1) {
      res->end(s.data(), s.length());
    } else {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}
















































































