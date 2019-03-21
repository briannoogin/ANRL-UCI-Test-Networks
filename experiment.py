from ActiveGuard import define_active_guard_model_with_connections
from FixedGuard import define_model_with_nofogbatchnorm_connections_extrainput,define_fixed_guard_baseline_model
from Baseline import define_baseline_functional_model
from loadData import load_data
from sklearn.model_selection import train_test_split
from FailureIteration import run
import keras.backend as K
import datetime
import os

# function to return average of a list 
def average(list):
    return sum(list) / len(list)
# runs all 3 failure configurations for all 3 models
if __name__ == "__main__":
    data,labels= load_data('mHealth_complete.log')
    training_data, test_data, training_labels, test_labels = train_test_split(data,labels,random_state = 42, test_size = .2, shuffle = True,stratify = labels)
    num_vars = len(training_data[0])
    num_classes = 13
    survive_configurations = [
        [.78,.8,.85],
        [.87,.91,.95],
        [.92,.96,.99]
    ]
    # survibility configurations for active guard basleline
    activeguard_baseline_surviveconfigs = [
        [.9,.9,.9],
        [.8,.8,.8],
        [.7,.7,.7],
        [.6,.6,.6],
        [.5,.5,.5],
    ]
    hidden_units = 250
    load_model = False
    now = datetime.datetime.now()
    date = str(now.month) + '-' + str(now.day) + '-' + str(now.year)
    file_name = 'results/' + date + '/results.txt'
    num_iterations = 2
    
    # declare list to keep track of accuracy across iterations
    active_guard_list = [0] * num_iterations
    fixed_guard_list = [0] * num_iterations
    baseline_list = [0] * num_iterations
    baseline_active_guard_list = [0] * num_iterations
    baseline_fixed_guard_list = [0] * num_iterations

    # keep track of output so that output is in order
    output_list = []
    for iteration in range(1,num_iterations+1):   
        with open(file_name,'a+') as file:
            output_list.append('ITERATION ' + str(iteration) +  '\n')
            file.write('ITERATION ' + str(iteration) +  '\n')
            print("ITERATION ", iteration)
        for survive_configuration in survive_configurations:
            K.set_learning_phase(1)
            # create models

            # active guard
            active_guard = define_active_guard_model_with_connections(num_vars,num_classes,hidden_units,0,survive_configuration)
            active_guard_file = 'models/no he_normal/' + str(iteration) + " " + str(survive_configuration) + ' new_active_guard.h5'
            if load_model:
                active_guard.load_weights(active_guard_file)
            else:
                active_guard.fit(data,labels,epochs=10, batch_size=128,verbose=1,shuffle = True)
                active_guard.save_weights(active_guard_file)

            # fixed guard
            fixed_guard = define_model_with_nofogbatchnorm_connections_extrainput(num_vars,num_classes,hidden_units,0,survive_configuration)
            fixed_guard_file = 'models/no he_normal/' + str(iteration) + " " +str(survive_configuration) + ' new_fixed_guard.h5'
            if load_model:
                fixed_guard.load_weights(fixed_guard_file)
            else:
                fixed_guard.fit(data,labels,epochs=10, batch_size=128,verbose=1,shuffle = True)
                fixed_guard.save_weights(fixed_guard_file)

            # fixed guard baseline
            baseline_fixed_guard = define_fixed_guard_baseline_model(num_vars,num_classes,hidden_units,0,survive_configuration)
            baseline_fixed_guard_file = 'models/no he_normal/' + str(iteration) + " " + str(survive_configuration) + ' baseline_fixed_guard.h5'
            if load_model:
                baseline_fixed_guard.load_weights(baseline_fixed_guard_file)
            else:
                baseline_fixed_guard.fit(data,labels,epochs=10, batch_size=128,verbose=1,shuffle = True)
                baseline_fixed_guard.save_weights(baseline_fixed_guard_file)

            # baseline model
            baseline = define_baseline_functional_model(num_vars,num_classes,hidden_units,0)
            baseline_file = 'models/no he_normal/' + str(iteration) + " " + str(survive_configuration) + ' new_baseline.h5'
            if load_model:
                baseline.load_weights(baseline_file)
            else:
                baseline.fit(data,labels,epochs=10, batch_size=128,verbose=1,shuffle = True)
                baseline.save_weights(baseline_file)
        
            # test models
            K.set_learning_phase(0)

            # make folder for outputs 
            if not os.path.exists('results/' + date):
                os.mkdir('results/' + date)

            # write results to a file 
            with open(file_name,'a+') as file:
                # survival configurations
                print(survive_configuration)
                file.write(str(survive_configuration) + '\n')

                # active guard
                file.write('ACTIVE GUARD' + '\n')
                output_list.append('ACTIVE GUARD' + '\n')
                print("ACTIVE GUARD")
                active_guard_list[iteration] = run(file_name,active_guard,survive_configuration,output_list,training_labels,test_data,test_labels)
                
                # fixed guard
                file.write('FIXED GUARD' + '\n')
                output_list.append('FIXED GUARD' + '\n')
                print("FIXED GUARD")
                fixed_guard_list[iteration] = run(file_name,fixed_guard,survive_configuration,output_list,training_labels,test_data,test_labels)

                # baseline fixed guard
                file.write('BASELINE FIXED GUARD' + '\n')
                output_list.append('BASELINE FIXED GUARD' + '\n')
                print("BASELINE FIXED GUARD")
                baseline_fixed_guard_list[iteration] = run(file_name,baseline_fixed_guard,survive_configuration,output_list,training_labels,test_data,test_labels)

                # baseline
                file.write('BASELINE' + '\n')
                output_list.append('BASELINE' + '\n')                    
                print("BASELINE")
                baseline_list[iteration] = run(file_name,baseline,survive_configuration,output_list,training_labels,test_data,test_labels)
                
        # runs baseline for active guard
        with open(file_name,'a+') as file:
            file.write('ACTIVE GUARD BASELINE' + '\n')
        output_list.append('ACTIVE GUARD BASELINE' + '\n')                  
        print("ACTIVE GUARD BASELINE")
        for survive_configuration in activeguard_baseline_surviveconfigs:
            K.set_learning_phase(1)
            baseline_activeguard_file = 'models/no he_normal/' + str(iteration) + " " + str(survive_configuration) + ' baseline_active_guard.h5'
            baseline_active_guard_model = define_active_guard_model_with_connections(num_vars,num_classes,hidden_units,0,survive_configuration)
            if load_model:
                baseline_active_guard_model.load_weights(baseline_activeguard_file)
            else:
                baseline_active_guard_model.fit(data,labels,epochs=10, batch_size=128,verbose=1,shuffle = True)
                baseline_active_guard_model.save_weights(baseline_activeguard_file)
            # write results to a file 
            with open(file_name,'a+') as file:
                print(survive_configuration)
                file.write(str(survive_configuration)+ '\n')
                output_list.append(str(survive_configuration)+ '\n')
                baseline_active_guard_list[iteration] = run(file_name,baseline_active_guard_model,survive_configuration,output_list,training_labels,test_data,test_labels)
   
   # write average accuracies to a file 
    with open(file_name,'a+') as file:
        active_guard_acc = average(active_guard_list)
        fixed_guard_acc = average(fixed_guard_list)
        baseline_acc = average(baseline_list)
        baseline_active_guard_acc = average(baseline_active_guard_list)
        baseline_fixed_guard_acc = average(baseline_fixed_guard_list)

        file.write(str(active_guard_acc) + '\n')
        file.write(str(fixed_guard_acc) + '\n')
        file.write(str(baseline_acc) + '\n')         
        file.write(str(baseline_active_guard_acc) + '\n')   
        file.write(str(baseline_fixed_guard_acc) + '\n')

        output_list.append(str(active_guard_acc) + '\n')
        output_list.append(str(fixed_guard_acc) + '\n')
        output_list.append(str(baseline_acc) + '\n')
        output_list.append(str(baseline_active_guard_acc) + '\n')
        output_list.append(str(baseline_fixed_guard_acc) + '\n')

        print("ActiveGuard Accuracy:",active_guard_acc)
        print("FixedGuard Accuracy:",fixed_guard_acc)
        print("Baseline Accuracy:",baseline_acc)
        print("Baseline ActiveGuard Accuracy:",baseline_active_guard_acc)
        print("Baseline FixedGuard Accuracy:",baseline_fixed_guard_acc)

        file.writelines(output_list)