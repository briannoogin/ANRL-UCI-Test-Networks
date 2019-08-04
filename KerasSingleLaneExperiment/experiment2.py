from KerasSingleLaneExperiment.deepFogGuardPlus import define_deepFogGuardPlus
from KerasSingleLaneExperiment.deepFogGuard import define_deepFogGuard
from KerasSingleLaneExperiment.loadData import load_data
from sklearn.model_selection import train_test_split
from KerasSingleLaneExperiment.FailureIteration import run
from KerasSingleLaneExperiment.main import average
import keras.backend as K
import datetime
import os
import gc 
from keras.callbacks import ModelCheckpoint
import numpy as np
# runs all hyperconnection configurations for both deepFogGuard survival configurations
# sensitivity analysis 
def main():
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
    survive_configurations = [
        [.78,.8,.85],
        [.87,.91,.95],
        [.92,.96,.99],
        [1,1,1]
    ]
    hyperconnections = [
        [0,0,0],
        [1,0,0],
        [0,1,0],
        [0,0,1],
        [1,1,0],
        [1,0,1],
        [0,1,1],
        [1,1,1],
    ]
    hidden_units = 250
    batch_size = 1028
    load_model = False
    now = datetime.datetime.now()
    date = str(now.month) + '-' + str(now.day) + '-' + str(now.year)
    file_name = 'results/fixed_split_experiment2_results.txt'
    num_iterations = 10
    verbose = 2
    # keep track of output so that output is in order
    output_list = []
    # dictionary to store all the results
    output = {
        "deepFogGuard":
        {
            "[0.78, 0.8, 0.85]":
            {
                "[0, 0, 0]":[0] * num_iterations,
                "[1, 0, 0]":[0] * num_iterations,
                "[0, 1, 0]":[0] * num_iterations,
                "[0, 0, 1]":[0] * num_iterations,
                "[1, 1, 0]":[0] * num_iterations,
                "[1, 0, 1]":[0] * num_iterations,
                "[0, 1, 1]":[0] * num_iterations,
                "[1, 1, 1]":[0] * num_iterations
            },
            "[0.87, 0.91, 0.95]":
            {
                "[0, 0, 0]":[0] * num_iterations,
                "[1, 0, 0]":[0] * num_iterations,
                "[0, 1, 0]":[0] * num_iterations,
                "[0, 0, 1]":[0] * num_iterations,
                "[1, 1, 0]":[0] * num_iterations,
                "[1, 0, 1]":[0] * num_iterations,
                "[0, 1, 1]":[0] * num_iterations,
                "[1, 1, 1]":[0] * num_iterations
            },
            "[0.92, 0.96, 0.99]":
            {
                "[0, 0, 0]":[0] * num_iterations,
                "[1, 0, 0]":[0] * num_iterations,
                "[0, 1, 0]":[0] * num_iterations,
                "[0, 0, 1]":[0] * num_iterations,
                "[1, 1, 0]":[0] * num_iterations,
                "[1, 0, 1]":[0] * num_iterations,
                "[0, 1, 1]":[0] * num_iterations,
                "[1, 1, 1]":[0] * num_iterations
            },
            "[1, 1, 1]":
            {
                "[0, 0, 0]":[0] * num_iterations,
                "[1, 0, 0]":[0] * num_iterations,
                "[0, 1, 0]":[0] * num_iterations,
                "[0, 0, 1]":[0] * num_iterations,
                "[1, 1, 0]":[0] * num_iterations,
                "[1, 0, 1]":[0] * num_iterations,
                "[0, 1, 1]":[0] * num_iterations,
                "[1, 1, 1]":[0] * num_iterations
            },
        }
    }
    # make folder for outputs 
    if not os.path.exists('results/' + date):
        os.mkdir('results/')
        os.mkdir('results/' + date)
    for iteration in range(1,num_iterations+1):   
        output_list.append('ITERATION ' + str(iteration) +  '\n')
        print("ITERATION ", iteration)
        K.set_learning_phase(1)
        for hyperconnection in hyperconnections:
          
            # deepFogGuard
            deepFogGuard = define_deepFogGuard(num_vars,num_classes,hidden_units,[1,1,1],hyperconnection)
            deepFogGuard_file = str(iteration) + " " + str(hyperconnection) +  'fixed_split_deepFogGuard.h5'
            if load_model:
                deepFogGuard.load_weights(deepFogGuard_file)
            else:
                print("Training deepFogGuard")
                dFGCheckPoint = ModelCheckpoint(deepFogGuard_file, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
                deepFogGuard.fit(training_data,training_labels,epochs=25, batch_size=batch_size,verbose=verbose,shuffle = True, callbacks = [dFGCheckPoint], validation_data=(val_data,val_labels))
                # load weights from epoch with the highest val acc
                deepFogGuard.load_weights(deepFogGuard_file)

            # test models
            K.set_learning_phase(0)

            for survive_configuration in survive_configurations:
                # write results to a file 
                # survival configurations
                print(survive_configuration)
                output_list.append(str(survive_configuration) + '\n')

                # deepFogGuard
                output_list.append('deepFogGuard' + '\n')
                print("deepFogGuard")
                output["deepFogGuard"][str(survive_configuration)][str(hyperconnection)][iteration-1] = run(deepFogGuard,survive_configuration,output_list,training_labels,test_data,test_labels)

            # clear session to remove old graphs from memory so that subsequent training is not slower
            K.clear_session()
            gc.collect()
            del deepFogGuard

   # write average accuracies to a file 
    with open(file_name,'a+') as file:
        for survive_configuration in survive_configurations:
            output_list.append(str(survive_configuration) + '\n')
            for hyperconnection in hyperconnections:
                output_list.append(str(hyperconnection) + '\n')
                deepFogGuard_acc = average(output["deepFogGuard"][str(survive_configuration)][str(hyperconnection)])
                deepFogGuard_std = np.std(output["deepFogGuard"][str(survive_configuration)][str(hyperconnection)],ddof=1)
                # write to output list
                output_list.append(str(survive_configuration) + " " + str(hyperconnection) + " deepFogGuard Accuracy: " + str(deepFogGuard_acc) + '\n')
                print(str(survive_configuration),str(hyperconnection),"deepFogGuard Accuracy:",deepFogGuard_acc)
                output_list.append(str(survive_configuration) + " " + str(hyperconnection) + " deepFogGuard std: " + str(deepFogGuard_std) + '\n')
                print(str(survive_configuration),str(hyperconnection),"deepFogGuard std:",deepFogGuard_std)
        file.writelines(output_list)
        file.flush()
        os.fsync(file)
    print(output)
    if use_GCP:
        os.system('gsutil -m -q cp -r {} gs://anrl-storage/results/'.format(file_name))

if __name__ == "__main__":
    main()