# ANRL-UCI-Test-Networks

#How to run on Google Cloud:
Setup Google Cloud credentials using gcloud init, must have API access to Google AI Platform enabled on GCP account
Run sh gcloud_train.sh to start training on gcp servers, can specify main file to run based on experiment variable in shell script.
Run sh local_train.sh to run code locally

#Potential issues
Current issues are that you cannot run via python command because current python import setup is configured for gcp. 
In order to fix issue, you have to remove KerasSingleLaneExperiment from all the file imports. 
ex: from KerasSingleLaneExperiment.cnn_deepFogGuard import define_deepFogGuard_CNN to from cnn_deepFogGuard import define_deepFogGuard_CNN
