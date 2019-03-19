from ActiveGuard import define_active_guard_model_with_connections
from FixedGuard import define_model_with_nofogbatchnorm_connections_extrainput,define_fixed_guard_baseline_model
from Baseline import define_baseline_functional_model
from loadData import load_data
from sklearn.model_selection import train_test_split
from FailureIteration import run
import keras.backend as K
import datetime
import os

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
        [.1,.1,.1],
        [.2,.2,.2],
        [.3,.3,.3],
        [.4,.4,.4],
        [.5,.5,.5],
    ]
    hidden_units = 250
    load_model = True
    now = datetime.datetime.now()
    date = str(now.month) + '-' + str(now.day) + '-' + str(now.year)
    file_name = 'results/' + date + '/results.txt'
    for survive_configuration in survive_configurations:
        K.set_learning_phase(1)
        # create models

        # active guard
        active_guard = define_active_guard_model_with_connections(num_vars,num_classes,hidden_units,0,survive_configuration)
        active_guard_file = 'models/' + str(survive_configuration) + ' new_active_guard.h5'
        if load_model:
            active_guard.load_weights(active_guard_file)
        else:
            active_guard.fit(data,labels,epochs=10, batch_size=128,verbose=1,shuffle = True)
            active_guard.save_weights(active_guard_file)

        # fixed guard
        fixed_guard = define_model_with_nofogbatchnorm_connections_extrainput(num_vars,num_classes,hidden_units,0,survive_configuration)
        fixed_guard_file = 'models/' + str(survive_configuration) + ' new_fixed_guard.h5'
        if load_model:
            fixed_guard.load_weights(fixed_guard_file)
        else:
            fixed_guard.fit(data,labels,epochs=10, batch_size=128,verbose=1,shuffle = True)
            fixed_guard.save_weights(fixed_guard_file)

        # fixed guard baseline
        baseline_fixed_guard = define_fixed_guard_baseline_model(num_vars,num_classes,hidden_units,0,survive_configuration)
        baseline_fixed_guard_file = 'models/' + str(survive_configuration) + ' baseline_fixed_guard.h5'
        if load_model:
            baseline_fixed_guard.load_weights(fixed_guard_file)
        else:
            baseline_fixed_guard.fit(data,labels,epochs=10, batch_size=128,verbose=1,shuffle = True)
            baseline_fixed_guard.save_weights(baseline_fixed_guard_file)

        # baseline model
        baseline = define_baseline_functional_model(num_vars,num_classes,hidden_units,0)
        baseline_file = 'models/' + str(survive_configuration) + ' new_baseline.h5'
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
            print(survive_configuration)
            file.write(str(survive_configuration) + '\n')
            file.write('ACTIVE GUARD' + '\n')
            print("ACTIVE GUARD")
            run(file_name,active_guard,survive_configuration,training_labels,test_data,test_labels)
            file.write('FIXED GUARD' + '\n')
            print("FIXED GUARD")
            run(file_name,fixed_guard,survive_configuration,training_labels,test_data,test_labels)
            file.write('BASELINE' + '\n')
            print("BASELINE")
            run(file_name,baseline,survive_configuration,training_labels,test_data,test_labels)
    # runs baseline for active guard
    with open(file_name,'a+') as file:
        print("ACTIVE GUARD BASELINE")
        file.write('ACTIVE GUARD BASELINE' + '\n')
    for survive_configuration in activeguard_baseline_surviveconfigs:
        K.set_learning_phase(1)
        baseline_activeguard_file = 'models/' + str(survive_configuration) + ' baseline_active_guard.h5'
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
            run(file_name,baseline_active_guard_model,survive_configuration,training_labels,test_data,test_labels)