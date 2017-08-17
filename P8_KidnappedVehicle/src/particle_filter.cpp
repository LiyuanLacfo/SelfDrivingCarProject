#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <limits>

#include "particle_filter.h"

using namespace std;

//declare the random number engin once for the life time of the program
static std::default_random_engine gen;//create a random number generator

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    num_particles = 100; //set the number of particles
    double std_x = std[0]; //standard deviation of x
    double std_y = std[1]; //standard deviation of y
    double std_theta = std[2]; //standard deviation of yaw

    //create normal distribution of x, y and theta
    std::normal_distribution<double> dist_x(x, std_x);
    std::normal_distribution<double> dist_y(y, std_y);
    std::normal_distribution<double> dist_theta(theta, std_theta);

    for(int i = 0; i < num_particles; i++){
        double x = dist_x(gen);
        double y = dist_y(gen);
        double theta = dist_theta(gen);
        double weight = 1.0;
        int id = i;
        Particle p;
        p.id = id;
        p.x = x;
        p.y = y;
        p.theta = theta;
        p.weight = weight;
        particles.push_back(p);
    }
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // TODO: Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/
    const double EPS = 0.001; //If the absolute value of yaw_rate is less that EPS, we see it as zero
    //create normal distribution for x, y and theta
    std::normal_distribution<double> dist_x(0, std_pos[0]);
    std::normal_distribution<double> dist_y(0, std_pos[1]);
    std::normal_distribution<double> dist_theta(0, std_pos[2]);

    if(fabs(yaw_rate) < EPS){
        for(int i = 0; i < num_particles; i++){
            double theta0 = particles[i].theta;
            double x0 = particles[i].x;
            double y0 = particles[i].y;
            //noise
            double noise_x = dist_x(gen);
            double noise_y = dist_y(gen);
            double noise_theta = dist_theta(gen);
            particles[i].x = x0 + velocity*delta_t*cos(theta0) + noise_x;
            particles[i].y = y0 + velocity*delta_t*sin(theta0) + noise_y;
            particles[i].theta = theta0  + noise_theta;
        }
    }
    else{
        for(int i = 0; i < num_particles; i++){
            double theta0 = particles[i].theta;
            double x0 = particles[i].x;
            double y0 = particles[i].y;
            //noise
            double noise_x = dist_x(gen);
            double noise_y = dist_y(gen);
            double noise_theta = dist_theta(gen);
            particles[i].x = x0 + velocity/yaw_rate*(sin(theta0+yaw_rate*delta_t)-sin(theta0)) + noise_x;
            particles[i].y = y0 + velocity/yaw_rate*(-cos(theta0+yaw_rate*delta_t)+cos(theta0)) + noise_y;
            particles[i].theta = theta0 + yaw_rate*delta_t + noise_theta;
        }
    }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
    for(size_t i = 0; i < observations.size(); i++){
        double min_dist = std::numeric_limits<double>::max();
        int target_id = -1;
        double obs_x = observations[i].x;
        double obs_y = observations[i].y;
        for(size_t j = 0; j < predicted.size(); j++){
            double pred_x = predicted[j].x;
            double pred_y = predicted[j].y;
            int pred_id = predicted[j].id;
            double d = dist(obs_x, obs_y, pred_x, pred_y);
            if(d < min_dist){
                min_dist = d;
                target_id = pred_id;
            }
        }
        observations[i].id = target_id;
    }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
    double std_x = std_landmark[0];
    double std_y = std_landmark[1];
    for(int i = 0; i < num_particles; i++){
        //get the coordinate of particle wrt map
        double p_x = particles[i].x;
        double p_y = particles[i].y;
        //the predicted landmarks for this particle
        std::vector<LandmarkObs> predicted;
        for(size_t j = 0; j < map_landmarks.landmark_list.size(); j++){
            double l_x = (double)map_landmarks.landmark_list[j].x_f;
            double l_y = (double)map_landmarks.landmark_list[j].y_f;
            int l_id = map_landmarks.landmark_list[j].id_i;
            double d = dist(p_x, p_y, l_x, l_y);
            if(d < sensor_range){
                LandmarkObs lm;
                lm.x = l_x;
                lm.y = l_y;
                lm.id = l_id;
                predicted.push_back(lm);
            }
        }
        std::vector<LandmarkObs> obs_landmarks;
        for(size_t h = 0; h < observations.size(); h++){
            double obs_x = observations[h].x;
            double obs_y = observations[h].y;
            double origin_x = particles[i].x;
            double origin_y = particles[i].y;
            double origin_theta = particles[i].theta;
            double new_x = origin_x + (obs_x*cos(origin_theta)-obs_y*sin(origin_theta));
            double new_y = origin_y + (obs_x*sin(origin_theta)+obs_y*cos(origin_theta));
            LandmarkObs obs_landmark;
            obs_landmark.x = new_x;
            obs_landmark.y = new_y;
            obs_landmarks.push_back(obs_landmark);
        }
        dataAssociation(predicted, obs_landmarks);
        //calculate new weight
        double weight = 1.0;
        for(size_t k = 0; k < obs_landmarks.size(); k++){
            double mu_x;
            double mu_y;
            for(size_t u = 0; u < map_landmarks.landmark_list.size(); u++){
                if(obs_landmarks[k].id == map_landmarks.landmark_list[u].id_i){
                    mu_x = (double)map_landmarks.landmark_list[u].x_f;
                    mu_y = (double)map_landmarks.landmark_list[u].y_f;
                    break;
                }
            }
            double o_x = obs_landmarks[k].x;
            double o_y = obs_landmarks[k].y;
            double norm_factor = 2*M_PI*std_x*std_y;
            //exp(-((x-mu_x)**2/(2*std_x**2)+(y-mu_y)**2/(2*std_y**2)))
            double p = exp(-(pow(o_x-mu_x, 2)/(2*pow(std_x, 2))+pow(o_y-mu_y, 2)/(2*pow(std_y, 2))))/norm_factor;
            weight *= p;
        }
        particles[i].weight = weight;
    }
    //normalize weight
    double sum = 0.0;
    for(int i = 0; i < num_particles; i++)
        sum += particles[i].weight;
    for(int i = 0; i < num_particles; i++)
        particles[i].weight /= (sum + std::numeric_limits<double>::epsilon());
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    std::vector<double> cur_weights;
    for(size_t i = 0; i < particles.size(); i++)
        cur_weights.push_back(particles[i].weight);
    std::discrete_distribution<int> dist_disc(cur_weights.begin(), cur_weights.end());
    std::vector<Particle> resampled_particles;
    for(size_t i = 0; i < particles.size(); i++){
        int ind = dist_disc(gen);
        resampled_particles.push_back(particles[ind]);
    }
    particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}