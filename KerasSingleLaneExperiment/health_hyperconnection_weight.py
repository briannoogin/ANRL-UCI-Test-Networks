
from KerasSingleLaneExperiment.deepFogGuard import define_deepFogGuard
from KerasSingleLaneExperiment.loadData import load_data
from sklearn.model_selection import train_test_split
from KerasSingleLaneExperiment.FailureIteration import calculateExpectedAccuracy
from KerasSingleLaneExperiment.main import average
import keras.backend as K
import gc
import os
from keras.callbacks import ModelCheckpoint

# runs all 3 failure configurations for all 3 models
if __name__ == "__main__":
    use_GCP = True
    if use_GCP == True:
        os.system('gsutil -m cp -r gs://anrl-storage/data/mHealth_complete.log ./')
        os.mkdir('models/')
    data,labels= load_data('mHealth_complete.log')
    # split data into train, val, and test
    # 80/10/10 split
    training_data, test_data, training_labels, test_labels = train_test_split(data,labels,random_state = 42, test_size = .20, shuffle = True)
    val_data, test_data, val_labels, test_labels = train_test_split(test_data,test_labels,random_state = 42, test_size = .50, shuffle = True)
    num_vars = len(training_data[0])
    num_classes = 13
    survivability_settings = [
        [1,1,1],
        [.92,.96,.99],
        [.87,.91,.95],
        [.78,.8,.85],
    ]
    hidden_units = 250
    batch_size = 1028
    load_model = False
    num_train_epochs = 25 
    # file name with the experiments accuracy output
    output_name = "results/health_hyperconnection_weight.txt"
    num_iterations = 10
    verbose = 2
    hyperconnection_weightedbysurvivability_config = 2
    # keep track of output so that output is in order
    output_list = []
    
    # convert survivability settings into strings so it can be used in the dictionary as keys
    no_failure = str(survivability_settings[0])
    normal = str(survivability_settings[1])
    poor = str(survivability_settings[2])
    hazardous = str(survivability_settings[3])

    # dictionary to store all the results
    output = {
        "DeepFogGuard Hyperconnection Weight": 
        {
            no_failure: [0] * num_iterations,
            hazardous:[0] * num_iterations,
            poor:[0] * num_iterations,
            normal:[0] * num_iterations,
        },
    }

    # make folder for outputs 
    if not os.path.exists('results/'):
        os.mkdir('results/')
    for iteration in range(1,num_iterations+1):   
        output_list.append('ITERATION ' + str(iteration) +  '\n')
        print("ITERATION ", iteration)

        for survivability_setting in survivability_settings:
            # deepFogGuard hyperconnection weight 
            deepFogGuard_hyperconnection_weight = define_deepFogGuard(num_vars,num_classes,hidden_units,survivability_setting, weight_config = hyperconnection_weightedbysurvivability_config)
            deepFogGuard_hyperconnection_weight_file = str(iteration) + " " + str(survivability_setting) + 'health_hyperconnection_weight.h5'
            if load_model:
                deepFogGuard_hyperconnection_weight.load_weights(deepFogGuard_hyperconnection_weight_file)
            else:
                print("DeepFogGuard Hyperconnection Weight")
                deepFogGuard_hyperconnection_weight_CheckPoint = ModelCheckpoint(deepFogGuard_hyperconnection_weight_file, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
                deepFogGuard_hyperconnection_weight.fit(training_data,training_labels,epochs=num_train_epochs, batch_size=batch_size,verbose=verbose,shuffle = True, callbacks = [deepFogGuard_hyperconnection_weight_CheckPoint],validation_data=(val_data,val_labels))
                # load weights from epoch with the highest val acc
                deepFogGuard_hyperconnection_weight.load_weights(deepFogGuard_hyperconnection_weight_file)

        # clear session so that model will recycled back into memory
        K.clear_session()
        gc.collect()
        del deepFogGuard_hyperconnection_weight
   # calculate average accuracies 
    for survivability_setting in survivability_settings:
        deepFogGuard_hyperconnection_weight_acc = average(output["DeepFogGuard Hyperconnection Weight"][str(survivability_setting)])
        output_list.append(str(survivability_setting) + " DeepFogGuard Hyperconnection Weight: " + str(deepFogGuard_hyperconnection_weight_acc) + '\n')
        print(str(survivability_setting),"DeepFogGuard Hyperconnection Weight:",deepFogGuard_hyperconnection_weight_acc)
    # write experiments output to file
    with open(output_name,'w') as file:
        file.writelines(output_list)
        file.flush()
        os.fsync(file)
    if use_GCP:
        os.system('gsutil -m -q cp -r {} gs://anrl-storage/results/'.format(output_name))
    print(output)
