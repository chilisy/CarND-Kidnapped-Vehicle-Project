/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
    //   x, y, theta and their uncertainties from GPS) and all weights to 1.
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    num_particles = 1000;
    
    Particle ptTemp;
    default_random_engine gen;
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);
    
    // loop over all particles and init
    for (int i=0; i<num_particles; i++)
    {
        ptTemp.x = dist_x(gen);
        ptTemp.y = dist_y(gen);
        ptTemp.theta = dist_theta(gen);
        ptTemp.weight = 1.0;
        ptTemp.id = i;
        
        weights.push_back(1.0);
        particles.push_back(ptTemp);
        
        //cout << ptTemp.x << ", " << ptTemp.y << ", " << ptTemp.theta << endl;
    }
    
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // TODO: Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/
    default_random_engine gen;
    normal_distribution<double> dist_x(0, std_pos[0]);
    normal_distribution<double> dist_y(0, std_pos[1]);
    normal_distribution<double> dist_theta(0, std_pos[2]);
    
    double c1 = velocity/yaw_rate;
    double c2 = yaw_rate*delta_t;
    
    double tol = 1e-6;
    
    
    for (int i=0; i<num_particles; i++)
    {
        double theta_0 = particles[i].theta;
        if(fabs(yaw_rate) > tol)
        {
            particles[i].x += c1*(sin(theta_0+c2) - sin(theta_0)) + dist_x(gen);
            particles[i].y += c1*(cos(theta_0) - cos(theta_0+c2)) + dist_y(gen);
            particles[i].theta += c2 + dist_theta(gen);
        }
        else
        {
            particles[i].x += velocity*delta_t*sin(theta_0) + dist_x(gen);
            particles[i].y += velocity*delta_t*cos(theta_0) + dist_y(gen);
            particles[i].theta += dist_theta(gen);
        }
        
        cout << particles[i].x << ", " << particles[i].y << ", " << particles[i].theta << endl;
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
    //   implement this method and use it as a helper during the updateWeights phase.
    
    
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
    
    
    
    // loop over all particles
    for(int i=0; i<num_particles; i++)
    {
        vector<LandmarkObs> obs_local = observations;
        
        for(int k=0; k<obs_local.size(); k++)
        {
            // transform to map coordinate system
            transform2MapCoord(obs_local[k], particles[i]);
            
        }
        
        // associate observations to map landmarks
        vector<int> associations;
        vector<double> obs_x;
        vector<double> obs_y;
        
        for(int k=0; k<obs_local.size(); k++)
        {
            int id = findAssociatedLandmark(obs_local[k], map_landmarks);
            associations.push_back(id);
            obs_x.push_back(obs_local[k].x);
            obs_y.push_back(obs_local[k].y);
            
        }
        
        particles[i] = SetAssociations(particles[i], associations, obs_x, obs_y);
        
        // calculate weights
        vector<double> weights;
        double weight = 1.0;
        
        weights = calculateWeights(particles[i], map_landmarks, std_landmark);
        
        particles[i].weight = accumulate(weights.begin(), weights.end(), 1.0, multiplies<double>());
        
        for(int k=0; k<weights.size(); k++)
            weight *= weights[k];
        
        cout << scientific << particles[i].weight << endl;
        
    }
    
    
}

vector<double> ParticleFilter::calculateWeights(Particle particle, Map map_landmarks, double std_landmark[])
{
    vector<double> weights;
    double std_x = std_landmark[0];
    double std_y = std_landmark[1];
    
    vector<Map::single_landmark_s> landmarkList = map_landmarks.landmark_list;
    
    for(int k=0; k<particle.associations.size(); k++)
    {
        int id = particle.associations[k];
        double x = landmarkList[id].x_f;
        double y = landmarkList[id].y_f;
        double mu_x = particle.sense_x[k];
        double mu_y = particle.sense_y[k];
        
        double p = exp(-(pow(x-mu_x, 2)/2.0/pow(std_x, 2) + pow(y-mu_y, 2)/2.0/pow(std_y, 2)));
        p = p/2.0/M_PI/std_x/std_y;
        
        weights.push_back(p);
    }
    
    return weights;
}

int ParticleFilter::findAssociatedLandmark(LandmarkObs obs, Map map_landmarks)
{
    int id;
    vector<Map::single_landmark_s> landmarkList = map_landmarks.landmark_list;
    vector<double> distances;
    
    for(int i=0; i<landmarkList.size(); i++)
    {
        double dist = sqrt(pow(obs.x-landmarkList[i].x_f, 2) + pow(obs.y-landmarkList[i].y_f, 2));
        distances.push_back(dist);
    }
    id = distance(distances.begin(),min_element(distances.begin(),distances.end()));
    
    return id;
}

void ParticleFilter::transform2MapCoord(LandmarkObs &obs, Particle particle)
{
    
    double theta_0 = particle.theta;
    double x_t = particle.x;
    double y_t = particle.y;
    double x = obs.x;
    double y = obs.y;
    
    obs.x = x*cos(theta_0) - y*sin(theta_0) + x_t;
    obs.y = x*sin(theta_0) + y*cos(theta_0) + y_t;
    
}

void ParticleFilter::resample() {
    // TODO: Resample particles with replacement with probability proportional to their weight.
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    
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
