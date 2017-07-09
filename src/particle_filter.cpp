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
#include <map>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
    //   x, y, theta and their uncertainties from GPS) and all weights to 1.
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    num_particles = 50;
    
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
    //cout << "prediction starts..." << endl;
    
    default_random_engine gen;
    normal_distribution<double> dist_x(0.0, std_pos[0]);
    normal_distribution<double> dist_y(0.0, std_pos[1]);
    normal_distribution<double> dist_theta(0.0, std_pos[2]);
    
    double c1 = velocity/yaw_rate;
    double c2 = yaw_rate*delta_t;
    
    double tol = 1e-5;
    
    
    for (int i=0; i<num_particles; i++)
    {
        double theta_0 = particles[i].theta;
        if(fabs(yaw_rate) > tol)
        {
            particles[i].x += c1*(sin(theta_0+c2) - sin(theta_0)) + dist_x(gen);
            particles[i].y += c1*(cos(theta_0) - cos(theta_0+c2)) + dist_y(gen);
            particles[i].theta += c2 + dist_theta(gen);
            
            /*
            while (particles[i].theta > M_PI  || particles[i].theta < -M_PI)
            {
                if (particles[i].theta > M_PI)
                    particles[i].theta -= 2*M_PI;
                else
                    particles[i].theta += 2*M_PI;
            }
            */
        }
        else
        {
            particles[i].x += velocity*delta_t*cos(theta_0) + dist_x(gen);
            particles[i].y += velocity*delta_t*sin(theta_0) + dist_y(gen);
            particles[i].theta += dist_theta(gen);
        }
        
        //cout << particles[i].x << ", " << particles[i].y << ", " << particles[i].theta << endl;
    }
    //cout << "prediction ends..." << endl;
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> pred_landmarks, std::vector<LandmarkObs>& observations) {
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
    //   implement this method and use it as a helper during the updateWeights phase.
    
    
    for(int i=0; i<observations.size(); i++)
    {
        vector<double> distances;
        
        for (int j=0; j<pred_landmarks.size(); j++)
        {
            double dist_val = dist(observations[i].x, observations[i].y,
                                   pred_landmarks[j].x, pred_landmarks[j].y);

            distances.push_back(dist_val);
        }
        
        int id = distance(distances.begin(),min_element(distances.begin(),distances.end()));
        observations[i].id = pred_landmarks[id].id;
        
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
    
    //cout << "updateWeights starts..." << endl;
    
    // loop over all particles
    for(int i=0; i<num_particles; i++)
    {
        vector<LandmarkObs> obs_local = observations;
        
        for(int k=0; k<obs_local.size(); k++)
        {
            // transform to map coordinate system
            transform2MapCoord(obs_local[k], particles[i]);
            
        }
        
        // obtain a list of landmarks which are in sensor range
        vector<LandmarkObs> pred_landmarks;
        std::map<int, Map::single_landmark_s> mappingLandmarks;
        
        for(int k=0; k<map_landmarks.landmark_list.size(); k++)
        {
            Map::single_landmark_s lan = map_landmarks.landmark_list[k];
            double distance = dist(lan.x_f, lan.y_f, particles[i].x, particles[i].y);
            if (distance <= sensor_range)
            {
                pred_landmarks.push_back(LandmarkObs{lan.id_i,lan.x_f,lan.y_f});
                // log the landmark so that it can be used later
                mappingLandmarks.insert(make_pair(lan.id_i, lan));
            }
        }
        
        if (pred_landmarks.size() > 0)
        {
            
            dataAssociation(pred_landmarks, obs_local);
            
            // calculate weights
            vector<double> weights;
            
            weights = calculateWeights(particles[i], obs_local, std_landmark, mappingLandmarks);
            
            particles[i].weight = accumulate(weights.begin(), weights.end(), 1.0, multiplies<double>());
            
        }
        
        //cout << scientific << particles[i].weight << endl;
        
    }
    
    //cout << "updateWeights ends..." << endl;
    
}

vector<double> ParticleFilter::calculateWeights(Particle particle, vector<LandmarkObs> observations, double std_landmark[], map<int, Map::single_landmark_s> mappingLandmarks)
{
    vector<double> weights;
    double std_x = std_landmark[0];
    double std_y = std_landmark[1];
    
    for(int k=0; k<observations.size(); k++)
    {
        int id = observations[k].id;
        //cout << mappingLandmarks[id].id_i << ", " << mappingLandmarks[id].x_f << endl;
        double x = mappingLandmarks[id].x_f;
        double y = mappingLandmarks[id].y_f;
        double mu_x = observations[k].x;
        double mu_y = observations[k].y;
        
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
    
    //cout << "resample starts..." << endl;
    
    vector<Particle> ptTemp;
    Particle pt2Add;
    vector<double> alpha_weights;
    
    vector<double> dist_weights(num_particles, 1.0);
    discrete_distribution<int> dist_fun(dist_weights.begin(), dist_weights.end());
    normal_distribution<double> dist(0.5, 0.5);
    default_random_engine gen;
    
    double mw;
    double beta;
    int index;
    
    // obtain weights of all particles
    for(int i=0; i<num_particles; i++)
    {
        weights[i] = particles[i].weight;
    }
    
    alpha_weights = normalize_vector(weights);
    
    mw = *max_element(alpha_weights.begin(), alpha_weights.end());
    index = dist_fun(gen);
    beta = 0.0;
    
    for(int i=0; i<num_particles; i++)
    {
        beta += 2.0 * mw * dist(gen);
        while (beta > alpha_weights[index])
        {
            beta -= alpha_weights[index];
            index = (index+1) % num_particles;
        }
        pt2Add = particles[index];
        pt2Add.id = i;
        ptTemp.push_back(pt2Add);
    }
    
    particles = ptTemp;
    
    
    
    //cout << "resample ends..." << endl;
    
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

//function to normalize a vector:
vector<double> ParticleFilter::normalize_vector(vector<double> inputVector){
    
    //declare sum:
    double sum = 0.0;
    
    //declare and resize output vector:
    std::vector<double> outputVector ;
    outputVector.resize(inputVector.size());
    
    //estimate the sum:
    for (unsigned int i = 0; i < inputVector.size(); ++i) {
        sum += inputVector[i];
    }
    
    //normalize with sum:
    for (unsigned int i = 0; i < inputVector.size(); ++i) {
        outputVector[i] = inputVector[i]/sum ;
    }
    
    //return normalized vector:
    return outputVector ;
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
