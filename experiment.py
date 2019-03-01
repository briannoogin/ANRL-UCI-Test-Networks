from ActiveGuard import define_active_guard_model_with_connections
from FixedGuard import define_model_with_nofogbatchnorm_connections_extrainput
from Baseline import define_baseline_functional_model
from loadData import load_data
from sklearn.model_selection import train_test_split
from FailureIteration import run
import keras.backend as K
# runs all 3 failure configurations for all 3 models
if __name__ == "__main__":
    data,labels= load_data('mHealth_complete.log')
    training_data, test_data, training_labels, test_labels = train_test_split(data,labels,random_state = 42, test_size = .2, shuffle = True,stratify = labels)
    num_vars = len(training_data[0])
    num_classes = 13
    survive_configurations = [
        [.78,8,.85],
        [.87,.91,.95],
        [.92,.96,.99]
    ]
    hidden_units = 250
    for survive_configuration in survive_configurations:
        # create models
        active_guard = define_active_guard_model_with_connections(num_vars,num_classes,hidden_units,0,survive_configuration)
        fixed_guard = define_model_with_nofogbatchnorm_connections_extrainput(num_vars,num_classes,hidden_units,0,survive_configuration)
        baseline = define_baseline_functional_model(num_vars,num_classes,hidden_units,0)

        # train models
        K.set_learning_phase(1)
        # active_guard.fit(data,labels,epochs=10, batch_size=128,verbose=1,shuffle = True)
        # fixed_guard.fit(data,labels,epochs=10, batch_size=128,verbose=1,shuffle = True)
        # baseline.fit(data,labels,epochs=10, batch_size=128,verbose=1,shuffle = True)

        # active_guard.save('models/active_guard.h5')
        # fixed_guard.save('models/fixed_guard.h5')
        # active_guard.save('models/active_guard.h5')

        # test models
        K.set_learning_phase(0)
        run(active_guard,survive_configuration,test_data,test_labels)
        run(fixed_guard,survive_configuration,test_data,test_labels)
        run(baseline,survive_configuration,test_data,test_labels)
