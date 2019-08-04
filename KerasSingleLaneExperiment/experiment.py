
from KerasSingleLaneExperiment.deepFogGuardPlus import define_deepFogGuardPlus
from KerasSingleLaneExperiment.deepFogGuard import define_deepFogGuard
from KerasSingleLaneExperiment.Vanilla import define_vanilla_model
from KerasSingleLaneExperiment.loadData import load_data
from sklearn.model_selection import train_test_split
from KerasSingleLaneExperiment.FailureIteration import run
from KerasSingleLaneExperiment.main import average
import keras.backend as K
import datetime
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
    survive_configurations = [
        [.78,.8,.85],
        [.87,.91,.95],
        [.92,.96,.99],
        [1,1,1]
    ]
    # survibility configurations for deepFogGuardPlus basleline
    deepFogGuardPlus_ablation_surviveconfigs = [
        [.95,.95,.95],
        [.9,.9,.9],
        [.7,.7,.7],
        [.5,.5,.5],
    ]
    hidden_units = 250
    batch_size = 1028
    load_model = False
    now = datetime.datetime.now()
    date = str(now.month) + '-' + str(now.day) + '-' + str(now.year)
    output_name = "results/newsplit_ablationHealthActivityExperiment_testwithsurvivability.txt"
    num_iterations = 10
    verbose = 2
    # keep track of output so that output is in order
    output_list = []
    # dictionary to store all the results
    output = {
        "deepFogGuard Plus":
        {
            "[0.78, 0.8, 0.85]":[0] * num_iterations,
            "[0.87, 0.91, 0.95]":[0] * num_iterations,
            "[0.92, 0.96, 0.99]":[0] * num_iterations,
            "[1, 1, 1]":[0] * num_iterations,
        }, 
        "deepFogGuard":
        {
            "[0.78, 0.8, 0.85]":[0] * num_iterations,
            "[0.87, 0.91, 0.95]":[0] * num_iterations,
            "[0.92, 0.96, 0.99]":[0] * num_iterations,
            "[1, 1, 1]":[0] * num_iterations,
        },
        "Vanilla": 
        {
            "[0.78, 0.8, 0.85]":[0] * num_iterations,
            "[0.87, 0.91, 0.95]":[0] * num_iterations,
            "[0.92, 0.96, 0.99]":[0] * num_iterations,
            "[1, 1, 1]":[0] * num_iterations,
        },
        "deepFogGuard Weight Ablation": 
        {
            "[0.78, 0.8, 0.85]":[0] * num_iterations,
            "[0.87, 0.91, 0.95]":[0] * num_iterations,
            "[0.92, 0.96, 0.99]":[0] * num_iterations,
            "[1, 1, 1]":[0] * num_iterations,
        },
        "deepFogGuard Plus Ablation": 
        {
             "[0.95, 0.95, 0.95]":
            {
                "[0.78, 0.8, 0.85]":[0] * num_iterations,
                "[0.87, 0.91, 0.95]":[0] * num_iterations,
                "[0.92, 0.96, 0.99]":[0] * num_iterations,
                "[1, 1, 1]":[0] * num_iterations,
            },
            "[0.9, 0.9, 0.9]":
            {
                "[0.78, 0.8, 0.85]":[0] * num_iterations,
                "[0.87, 0.91, 0.95]":[0] * num_iterations,
                "[0.92, 0.96, 0.99]":[0] * num_iterations,
                "[1, 1, 1]":[0] * num_iterations,
            },
            "[0.7, 0.7, 0.7]":
            {
                "[0.78, 0.8, 0.85]":[0] * num_iterations,
                "[0.87, 0.91, 0.95]":[0] * num_iterations,
                "[0.92, 0.96, 0.99]":[0] * num_iterations,
                "[1, 1, 1]":[0] * num_iterations,
            },
            "[0.5, 0.5, 0.5]":
            {
                "[0.78, 0.8, 0.85]":[0] * num_iterations,
                "[0.87, 0.91, 0.95]":[0] * num_iterations,
                "[0.92, 0.96, 0.99]":[0] * num_iterations,
                "[1, 1, 1]":[0] * num_iterations,
            },
        }
    }

    # make folder for outputs 
    if not os.path.exists('results/'):
        os.mkdir('results/')
    for iteration in range(1,num_iterations+1):   
        output_list.append('ITERATION ' + str(iteration) +  '\n')
        print("ITERATION ", iteration)
        K.set_learning_phase(1)
        # create models

        # deepFogGuardPlus
        deepFogGuardPlus = define_deepFogGuardPlus(num_vars,num_classes,hidden_units,[.95,.95,.95])
        deepFogGuardPlus_file = "new_split_" + str(iteration) + '_deepFogGuardPlus.h5'
        if load_model:
            deepFogGuardPlus.load_weights(deepFogGuardPlus_file)
        else:
            print("Training deepFogGuardPlus")
            dFGPlusCheckPoint = ModelCheckpoint(deepFogGuardPlus_file, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
            deepFogGuardPlus.fit(training_data,training_labels,epochs=25, batch_size=batch_size,verbose=verbose,shuffle = True, callbacks = [dFGPlusCheckPoint],validation_data=(val_data,val_labels))
            # load weights from epoch with the highest val acc
            deepFogGuardPlus.load_weights(deepFogGuardPlus_file)

        # deepFogGuard
        deepFogGuard = define_deepFogGuard(num_vars,num_classes,hidden_units,[1,1,1])
        deepFogGuard_file = "new_split_" + str(iteration) + '_deepFogGuard.h5'
        if load_model:
            deepFogGuard.load_weights(deepFogGuard_file)
        else:
            print("Training deepFogGuard")
            dFGCheckPoint = ModelCheckpoint(deepFogGuard_file, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
            deepFogGuard.fit(training_data,training_labels,epochs=25, batch_size=batch_size,verbose=verbose,shuffle = True, callbacks = [dFGCheckPoint], validation_data=(val_data,val_labels))
            # load weights from epoch with the highest val acc
            deepFogGuard.load_weights(deepFogGuard_file)


        # vanilla model
        vanilla = define_vanilla_model(num_vars,num_classes,hidden_units)
        vanilla_file = "new_split_" + str(iteration) + '_vanilla.h5'
        if load_model:
            vanilla.load_weights(vanilla_file)
        else:
            print("Training vanilla")
            vanillaCheckPoint = ModelCheckpoint(vanilla_file, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
            vanilla.fit(training_data,training_labels,epochs=25, batch_size=batch_size,verbose=verbose,shuffle = True, callbacks = [vanillaCheckPoint],validation_data=(val_data,val_labels))
            # load weights from epoch with the highest val acc
            vanilla.load_weights(vanilla_file)
 
        # test models
        K.set_learning_phase(0)

        for survive_configuration in survive_configurations:
            # deepFogGuard weight ablation
            deepFogGuard_weight_ablation = define_deepFogGuard(num_vars,num_classes,hidden_units,survive_configuration, weight_config = 2)
            deepFogGuard_weight_ablation_file = str(iteration) + " " + str(survive_configuration) + '_new_split_deepFogGuard_weight_ablation_testsurvivalrate.h5'
            if load_model:
                deepFogGuard_weight_ablation.load_weights(deepFogGuard_weight_ablation_file)
            else:
                print("Training deepFogGuard Weight Ablation")
                deepFogGuard_weight_ablation_CheckPoint = ModelCheckpoint(deepFogGuard_weight_ablation_file, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
                deepFogGuard_weight_ablation.fit(training_data,training_labels,epochs=25, batch_size=batch_size,verbose=verbose,shuffle = True, callbacks = [deepFogGuard_weight_ablation_CheckPoint],validation_data=(val_data,val_labels))
                # load weights from epoch with the highest val acc
                deepFogGuard_weight_ablation.load_weights(deepFogGuard_weight_ablation_file)

            # deepFogGuard Plus
            output_list.append('deepFogGuard Plus' + '\n')
            print("deepFogGuard Plus")
            output["deepFogGuard Plus"][str(survive_configuration)][iteration-1] = run(deepFogGuardPlus,survive_configuration,output_list,training_labels,test_data,test_labels)

            # deepFogGuard
            output_list.append('deepFogGuard' + '\n')
            print("deepFogGuard")
            output["deepFogGuard"][str(survive_configuration)][iteration-1] = run(deepFogGuard,survive_configuration,output_list,training_labels,test_data,test_labels)

            # deepFogGuard Weight Ablation
            output_list.append('deepFogGuard Weight Ablation' + '\n')
            print("deepFogGuard Weight Ablation")
            output["deepFogGuard Weight Ablation"][str(survive_configuration)][iteration-1] = run(deepFogGuard_weight_ablation,survive_configuration,output_list,training_labels,test_data,test_labels)

            # vanilla
            output_list.append('Vanilla' + '\n')                    
            print("Vanilla")
            output["Vanilla"][str(survive_configuration)][iteration-1] = run(vanilla,survive_configuration,output_list,training_labels,test_data,test_labels)

        #runs deepFogGuard Plus Ablation
        output_list.append('deepFogGuard Plus Ablation' + '\n')                  
        print("deepFogGuard Plus Ablation")
        for survive_configuration in deepFogGuardPlus_ablation_surviveconfigs:
            K.set_learning_phase(1)
            deepFogGuardPlus_Ablation_file = str(iteration) + " " + str(survive_configuration) + '_new_split_deepFogGuardPlus_Ablation.h5'
            deepFogGuardPlus_ablation = define_deepFogGuardPlus(num_vars,num_classes,hidden_units,survive_configuration)
            if load_model:
                deepFogGuardPlus_ablation.load_weights(deepFogGuardPlus_Ablation_file)
            else:
                print("Training deepFogGuard Plus Ablation")
                print(str(survive_configuration))
                deepFogGuardPlus_ablation_CheckPoint = ModelCheckpoint(deepFogGuardPlus_Ablation_file, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
                deepFogGuardPlus_ablation.fit(training_data,training_labels,epochs=25, batch_size=batch_size,verbose=verbose,shuffle = True, callbacks = [deepFogGuardPlus_ablation_CheckPoint],validation_data=(val_data,val_labels))
                deepFogGuardPlus_ablation.load_weights(deepFogGuardPlus_Ablation_file)
                print("Test on normal survival rates")
                output_list.append("Test on normal survival rates" + '\n')
                for normal_survival_config in survive_configurations:
                    output_list.append(str(normal_survival_config)+ '\n')
                    print(normal_survival_config)
                    output["deepFogGuard Plus Ablation"][str(survive_configuration)][str(normal_survival_config)][iteration-1] = run(deepFogGuardPlus_ablation,normal_survival_config,output_list,training_labels,test_data,test_labels)
            # clear session so that model will recycled back into memory
            K.clear_session()
            gc.collect()
            del deepFogGuardPlus_ablation
        # clear session so that model will recycled back into memory
        K.clear_session()
        gc.collect()
        del deepFogGuard
        del deepFogGuard_weight_ablation
        del deepFogGuardPlus
        del vanilla
   # calculate average accuracies 
    for survive_configuration in survive_configurations:
        deepfogGuardPlus_acc = average(output["deepFogGuard Plus"][str(survive_configuration)])
        deepFogGuard_acc = average(output["deepFogGuard"][str(survive_configuration)])
        vanilla_acc = average(output["Vanilla"][str(survive_configuration)])
        deepFogGuard_WeightAblation_acc = average(output["deepFogGuard Weight Ablation"][str(survive_configuration)])

        output_list.append(str(survive_configuration) + " deepFogGuard Plus Accuracy: " + str(deepfogGuardPlus_acc) + '\n')
        output_list.append(str(survive_configuration) + " deepFogGuard Accuracy: " + str(deepFogGuard_acc) + '\n')
        output_list.append(str(survive_configuration) + " Vanilla Accuracy: " + str(vanilla_acc) + '\n')
        output_list.append(str(survive_configuration) + " deepFogGuard Weight Ablation: " + str(deepFogGuard_WeightAblation_acc) + '\n')

        print(str(survive_configuration),"deepFogGuard Plus Accuracy:",deepfogGuardPlus_acc)
        print(str(survive_configuration),"deepFogGuard Accuracy:",deepFogGuard_acc)
        print(str(survive_configuration),"Vanilla Accuracy:",vanilla_acc)
        print(str(survive_configuration),"deepFogGuard Weight Ablation:",deepFogGuard_WeightAblation_acc)

    #calculate average accuracies for deepFogGuard Plus Ablation
    for survive_config in deepFogGuardPlus_ablation_surviveconfigs:
        print(survive_config)
        for original_survive_config in survive_configurations:
            deepFogGuardPlus_Ablation_acc = average(output["deepFogGuard Plus Ablation"][str(survive_config)][str(original_survive_config)])
            output_list.append(str(survive_config) + str(original_survive_config) + " deepFogGuard Plus Ablation: " + str(deepFogGuardPlus_Ablation_acc) + '\n')
            print(survive_config,original_survive_config,"deepFogGuard Plus Ablation:",deepFogGuardPlus_Ablation_acc)  
    with open(output_name,'w') as file:
        file.writelines(output_list)
        file.flush()
        os.fsync(file)
    if use_GCP:
        os.system('gsutil -m -q cp -r {} gs://anrl-storage/results/'.format(output_name))
    print(output)