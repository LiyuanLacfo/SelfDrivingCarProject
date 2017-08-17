#include <iostream>
#include "particle_filter.h"
#include <cmath>
#include <vector>

using namespace std;

int main()
{
    ParticleFilter pf;
    double x = 3;
    double y = 3;
    double theta = M_PI/2;
    double std[3] = {1, 2, 1};
    pf.init(x, y, theta, std);
    cout << pf.particles[0].x << endl;
    //predict
    double delta_t = 0.1;
    double std_pos[3] = {1, 1, 1};
    double velocity = 30;
    double yaw_rate = M_PI/10;
    pf.prediction(delta_t, std_pos, velocity, yaw_rate);
    cout << pf.particles[0].x << endl;
    pf.printOut();
    std::vector<LandmarkObs> predicted;
    std::vector<LandmarkObs> land_obs;
    //number of landmark
    int T = 2;
    for(int i = 0; i < T; i++){
        LandmarkObs l;
        LandmarkObs g;
        l.x = 1;
        l.y = 1;
        l.id = i+2;
        g.x = 2;
        g.y = 2;
        g.id = i;
        predicted.push_back(l);
        land_obs.push_back(g);
    }
    // pf.dataAssociation(predicted, land_obs);
    // for(int i = 0; i < T; i++)
    //     cout << land_obs[i].id << endl;
    //read map data
    Map map;
    std::string filename;
    filename = "../data/map_data.txt";
    read_map_data(filename, map);
    //updateWeights(double sensor_range, double std_landmark[], 
        //std::vector<LandmarkObs> observations, Map map_landmarks)
    double std_landmark[2] = {1, 1};
    cout << "The weight of first one: " << pf.particles[0].weight << endl;
    pf.updateWeights(50, std_landmark, land_obs, map);
    cout << "The weight of first one: " << pf.particles[0].weight << endl;
    pf.resample();
    cout << "The weight of first one: " << pf.particles[0].weight << endl;
    return 0;
}