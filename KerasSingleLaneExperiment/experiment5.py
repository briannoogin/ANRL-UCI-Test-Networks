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

# experiment with new active guard 
# do active guard results for everything (table 1, table 5, extra table at the end)

def normal_experiment():
    use_GCP = False
    if use_GCP == True:
        os.system('gsutil -m cp -r gs://anrl-storage/data/mHealth_complete.log ./')
    data,labels= load_data('mHealth_complete.log')
    training_data, test_data, training_labels, test_labels = train_test_split(data,labels,random_state = 42, test_size = .2, shuffle = True, stratify = labels)
    input_size = len(training_data[0])
    num_classes = 13
    hidden_units = 250
    batch_size = 1028
    verbose = 2
    output_list = []

    now = datetime.datetime.now()
    date = str(now.month) + '-' + str(now.day) + '-' + str(now.year)
    # make folder for outputs 
    # if not os.path.exists('results/' + date):
    #     os.mkdir('results/')
    #     os.mkdir('results/' + date)
    file_name = 'results/' + date + '/experiment5_95dropout_experiment_results.txt'
    survive_configurations = [
        [.78,.8,.85],
        [.87,.91,.95],
        [.92,.96,.99],
        [1,1,1]
    ]
    num_iterations = 10
    output = {
        "Active Guard":
        {
            "[0.78, 0.8, 0.85]": [0] * num_iterations,
            "[0.87, 0.91, 0.95]":[0] * num_iterations,
            "[0.92, 0.96, 0.99]": [0] * num_iterations,
            "[1, 1, 1]":[0] * num_iterations,
        },
    }
    for iteration in range(1,num_iterations+1):
        for survive_configuration in survive_configurations:
            model = define_deepFogGuardPlus(input_size,num_classes,hidden_units,[.95,.95,.95],[1,1,1])
            model.fit(data,labels,epochs=10, batch_size=batch_size,verbose=verbose,shuffle = True)
            output["Active Guard"][str(survive_configuration)][iteration-1] = run(model,survive_configuration,output_list,training_labels,test_data,test_labels)
            # clear session to remove old graphs from memory so that subsequent training is not slower
            K.clear_session()
            gc.collect()
            del model
        # no failure 
    
     # write average accuracies to a file 
    with open(file_name,'a+') as file:
        for survive_configuration in survive_configurations:
            output_list.append(str(survive_configuration) + '\n')
            active_guard_acc = average(output["Active Guard"][str(survive_configuration)])
            output_list.append(str(survive_configuration) + " ActiveGuard Accuracy: " + str(active_guard_acc) + '\n')
            print(str(survive_configuration),"ActiveGuard Accuracy:",active_guard_acc)
        file.writelines(output_list)
        file.flush()
        os.fsync(file)
    if use_GCP:
        os.system('gsutil -m -q cp -r {} gs://anrl-storage/results/'.format(file_name))

if __name__ == "__main__":
    normal_experiment()