'''
Created on May 12, 2015
'''
import os

sim_files_folder = "./Simulation_MAB_files"
save_address = "./SimulationResults"
LastFM_save_address = "./LastFMResults"
Delicious_save_address = "./DeliciousResults"
Yahoo_save_address = "./YahooResults"
MovieLens_save_address = './MovieLensResults'

save_addressResult = "./Results/Sparse"

datasets_address = '.'  # should be modified accoring to the local address

LastFM_address = datasets_address + '/Dataset/hetrec2011-lastfm-2k/processed_data'
Delicious_address = datasets_address + '/Dataset/hetrec2011-delicious-2k/processed_data'
MovieLens_address = datasets_address + '/Dataset/ml-20m/processed_data'

LastFM_FeatureVectorsFileName = os.path.join(LastFM_address, 'Arm_FeatureVectors_2.dat')
LastFM_relationFileName = os.path.join(LastFM_address, 'user_friends.dat.mapped')

Delicious_FeatureVectorsFileName = os.path.join(Delicious_address, 'Arm_FeatureVectors_2.dat')
Delicious_relationFileName = os.path.join(Delicious_address, 'user_contacts.dat.mapped')

MovieLens_FeatureVectorsFileName = os.path.join(MovieLens_address, 'Arm_FeatureVectors_2.dat')
MovieLens_relationFileName = os.path.join(MovieLens_address, 'user_contacts.dat.mapped')