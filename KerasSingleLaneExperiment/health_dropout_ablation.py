# survibility configurations for deepFogGuardPlus basleline
from KerasSingleLaneExperiment.deepFogGuardPlus import define_deepFogGuardPlus
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
    # survibility configurations for deepFogGuardPlus baseline
    ablation_survivalrate_configurations = [
        [.95,.95,.95],
        [.9,.9,.9],
        [.7,.7,.7],
        [.5,.5,.5],
    ]
    hidden_units = 250
    batch_size = 1028
    load_model = False
    num_train_epochs = 25 
    # file name with the experiments accuracy output
    output_name = "results/health_dropout_ablation.txt"
    num_iterations = 10
    verbose = 2
    # keep track of output so that output is in order
    output_list = []
    
    # convert survivability settings into strings so it can be used in the dictionary as keys
    no_failure = str(survivability_settings[0])
    normal = str(survivability_settings[1])
    poor = str(survivability_settings[2])
    hazardous = str(survivability_settings[3])

    # dictionary to store all the results
    output = {
        "deepFogGuard Plus Ablation": 
        {
             "[0.95, 0.95, 0.95]":
            {
                hazardous:[0] * num_iterations,
                poor:[0] * num_iterations,
                normal:[0] * num_iterations,
                no_failure:[0] * num_iterations,
            },
            "[0.9, 0.9, 0.9]":
            {
                hazardous:[0] * num_iterations,
                poor:[0] * num_iterations,
                normal:[0] * num_iterations,
                no_failure:[0] * num_iterations,
            },
            "[0.7, 0.7, 0.7]":
            {
                hazardous:[0] * num_iterations,
                poor:[0] * num_iterations,
                normal:[0] * num_iterations,
                no_failure:[0] * num_iterations,
            },
            "[0.5, 0.5, 0.5]":
            {
                hazardous:[0] * num_iterations,
                poor:[0] * num_iterations,
                normal:[0] * num_iterations,
                no_failure:[0] * num_iterations,
            },
        }
    }

    # make folder for outputs 
    if not os.path.exists('results/'):
        os.mkdir('results/')
    for iteration in range(1,num_iterations+1):   
        output_list.append('ITERATION ' + str(iteration) +  '\n')
        print("ITERATION ", iteration)
        output_list.append('deepFogGuard Plus Ablation' + '\n')                  
        print("deepFogGuard Plus Ablation")
        for nodewise_survival_rate in ablation_survivalrate_configurations:
            deepFogGuardPlus_Ablation_file = str(iteration) + " " + str(nodewise_survival_rate) + '_new_split_deepFogGuardPlus_Ablation.h5'
            deepFogGuardPlus_ablation = define_deepFogGuardPlus(num_vars,num_classes,hidden_units,nodewise_survival_rate)
            if load_model:
                deepFogGuardPlus_ablation.load_weights(deepFogGuardPlus_Ablation_file)
            else:
                print("Training deepFogGuard Plus Ablation")
                print(str(nodewise_survival_rate))
                deepFogGuardPlus_ablation_CheckPoint = ModelCheckpoint(deepFogGuardPlus_Ablation_file, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
                deepFogGuardPlus_ablation.fit(training_data,training_labels,epochs=num_train_epochs, batch_size=batch_size,verbose=verbose,shuffle = True, callbacks = [deepFogGuardPlus_ablation_CheckPoint],validation_data=(val_data,val_labels))
                deepFogGuardPlus_ablation.load_weights(deepFogGuardPlus_Ablation_file)
                print("Test on normal survival rates")
                output_list.append("Test on normal survival rates" + '\n')
                for survival_config in survivability_settings:
                    output_list.append(str(survival_config)+ '\n')
                    print(survival_config)
                    output["deepFogGuard Plus Ablation"][str(nodewise_survival_rate)][str(survival_config)][iteration-1] = calculateExpectedAccuracy(deepFogGuardPlus_ablation,survival_config,output_list,training_labels,test_data,test_labels)
            # clear session so that model will recycled back into memory
            K.clear_session()
            gc.collect()
            del deepFogGuardPlus_ablation

    # calculate average accuracies for deepFogGuard Plus Ablation
    for nodewise_survival_rate in ablation_survivalrate_configurations:
        print(nodewise_survival_rate)
        for survive_config in survivability_settings:
            deepFogGuardPlus_Ablation_acc = average(output["deepFogGuard Plus Ablation"][str(nodewise_survival_rate)][str(survive_config)])
            output_list.append(str(nodewise_survival_rate) + str(survive_config) + " deepFogGuard Plus Ablation: " + str(deepFogGuardPlus_Ablation_acc) + '\n')
            print(nodewise_survival_rate,survive_config,"deepFogGuard Plus Ablation:",deepFogGuardPlus_Ablation_acc)  
    # write experiments output to file
    with open(output_name,'w') as file:
        file.writelines(output_list)
        file.flush()
        os.fsync(file)
    if use_GCP:
        os.system('gsutil -m -q cp -r {} gs://anrl-storage/results/'.format(output_name))
    print(output)
