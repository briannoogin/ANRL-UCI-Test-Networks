
from keras.datasets import cifar10
from keras.applications.mobilenet import MobileNet
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import math
import os 
from KerasSingleLaneExperiment.cnn import define_Vanilla_CNN, define_deepFogGuard_CNN, define_deepFogGuardPlus_CNN
from KerasSingleLaneExperiment.FailureIteration import run
import numpy as np
from KerasSingleLaneExperiment.experiment import average
import datetime
import gc
from sklearn.model_selection import train_test_split

# Vanilla, deepFogGuard, and deepFogGuard+ experiments
def cnn_normal_experiments():
    # get cifar10 data 
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # normalize input
    x_train = x_train / 255
    x_test = x_test / 255
    # Concatenate train and test images
    x = np.concatenate((x_train,x_test))
    y = np.concatenate((y_train,y_test))

    # split data into train, validation, and holdout set (80/10/10)
    x_train, x_test, y_train, y_test = train_test_split(x,y,random_state = 42, test_size = .20, shuffle = True)
    x_val, x_test, y_val, y_test = train_test_split(x_test,y_test,random_state = 42, test_size = .50, shuffle = True)
    train_datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    )

    survive_configs = [
        [.96,.98],
        [.90,.95],
        [.80,.85],
        [1,1]
    ]
    num_iterations = 10
    output = {
        "deepFogGuard Plus":
        {
            "[0.96, 0.98]": [0] * num_iterations,
            "[0.9, 0.95]":[0] * num_iterations,
            "[0.8, 0.85]":[0] * num_iterations,
            "[1, 1]":[0] * num_iterations,
        },
        "deepFogGuard":
        {
            "[0.96, 0.98]": [0] * num_iterations,
            "[0.9, 0.95]":[0] * num_iterations,
            "[0.8, 0.85]":[0] * num_iterations,
            "[1, 1]":[0] * num_iterations,
        },
        "Vanilla":
        {
            "[0.96, 0.98]": [0] * num_iterations,
            "[0.9, 0.95]":[0] * num_iterations,
            "[0.8, 0.85]":[0] * num_iterations,
            "[1, 1]":[0] * num_iterations,
        },
    }
    now = datetime.datetime.now()
    date = str(now.month) + '-' + str(now.day) + '-' + str(now.year)
    # make folder for outputs 
    if not os.path.exists('results/'):
        os.mkdir('results/')
    if not os.path.exists('results/' + date):
        os.mkdir('results/' + date)
    if not os.path.exists('models'):      
        os.mkdir('models/')
    file_name = 'results/' + date + '/experiment3_fixedsplit_normalDeepFogGuardPlusExperiment_results.txt'
    output_list = []
    for iteration in range(1,num_iterations+1):
        print("iteration:",iteration)
        vanilla_name = "vanilla_cnn_fixedsplit_" + str(iteration) + ".h5"
        deepFogGuard_name = "deepFogGuard_cnn_fixedsplit_" + str(iteration) + ".h5"
        deepFogGuardPlus_name = "deepFogGuardPlus_cnn_fixedsplit" + str(iteration) + ".h5"

        vanilla = define_Vanilla_CNN(weights = None,classes=10,input_shape = (32,32,3),dropout = 0, alpha = .5)
        deepFogGuard = define_deepFogGuard_CNN(weights = None,classes=10,input_shape = (32,32,3),dropout = 0, alpha = .5)
        deepFogGuardPlus = define_deepFogGuardPlus_CNN(weights = None,classes=10,input_shape = (32,32,3),dropout = 0, alpha = .5,survive_rates=[.95,.95,.95])

        batch_size = 128
        train_steps_per_epoch = math.ceil(len(x_train) / batch_size)
        val_steps_per_epoch = math.ceil(len(x_val) / batch_size)

        # checkpoints to keep track of model with best validation accuracy 
        vanillaCheckPoint = ModelCheckpoint(vanilla_name, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
        deepFogGuardCheckPoint = ModelCheckpoint(deepFogGuard_name, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
        deepFogGuardPlusCheckPoint = ModelCheckpoint(deepFogGuardPlus_name, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)

        # fit cnns
        vanilla.fit_generator(
            train_datagen.flow(x_train,y_train,batch_size = batch_size),
            epochs = 75,
            validation_data = (x_val,y_val), 
            steps_per_epoch = train_steps_per_epoch, 
            verbose = 2, 
            validation_steps = val_steps_per_epoch,
            callbacks = [vanillaCheckPoint])
        deepFogGuard.fit_generator(
            train_datagen.flow(x_train,y_train,batch_size = batch_size),
            epochs = 75,
            validation_data = (x_val,y_val), 
            steps_per_epoch = train_steps_per_epoch, 
            verbose = 2,
            validation_steps = val_steps_per_epoch,
            callbacks = [deepFogGuardCheckPoint])
        deepFogGuardPlus.fit_generator(
            train_datagen.flow(x_train,y_train,batch_size = batch_size),
            epochs = 75,
            validation_data = (x_val,y_val),
            steps_per_epoch = train_steps_per_epoch,
            verbose = 2,
            validation_steps = val_steps_per_epoch,
            callbacks = [deepFogGuardPlusCheckPoint])

        # load weights with the highest val accuracy
        vanilla.load_weights(vanilla_name)
        deepFogGuard.load_weights(deepFogGuard_name)
        deepFogGuardPlus.load_weights(deepFogGuardPlus_name)

        for survive_config in survive_configs:
            output_list.append(str(survive_config) + '\n')
            print(survive_config)
            output["Vanilla"][str(survive_config)][iteration-1] = run(vanilla, survive_config,output_list, y_train, x_test, y_test)
            output["deepFogGuard"][str(survive_config)][iteration-1] = run(deepFogGuard, survive_config,output_list, y_train, x_test, y_test)
            output["deepFogGuard Plus"][str(survive_config)][iteration-1] = run(deepFogGuardPlus, survive_config,output_list, y_train, x_test, y_test)
        # clear session so that model will recycled back into memory
        K.clear_session()
        gc.collect()
        del vanilla
        del deepFogGuard
        del deepFogGuardPlus
    with open(file_name,'a+') as file:
        for survive_config in survive_configs:
            output_list.append(str(survive_config) + '\n')

            vanilla_acc = average(output["Vanilla"][str(survive_config)])
            deepFogGuard_acc = average(output["deepFogGuard"][str(survive_config)])
            deepFogGuardPlus_acc = average(output["deepFogGuard Plus"][str(survive_config)])

            output_list.append(str(survive_config) + " Vanilla Accuracy: " + str(vanilla_acc) + '\n')
            output_list.append(str(survive_config) + " deepFogGuard Accuracy: " + str(deepFogGuard_acc) + '\n')
            output_list.append(str(survive_config) + " deepFogGuard Plus Accuracy: " + str(deepFogGuardPlus_acc) + '\n')

            print(str(survive_config),"Vanilla Accuracy:",vanilla_acc)
            print(str(survive_config),"deepFogGuard Accuracy:",deepFogGuard_acc)
            print(str(survive_config),"deepFogGuard Plus Accuracy:",deepFogGuardPlus_acc)
        file.writelines(output_list)
        file.flush()
        os.fsync(file)
    use_GCP = True
    if use_GCP:
        os.system('gsutil -m -q cp -r *.h5 gs://anrl-storage/models')
        os.system('gsutil -m -q cp -r {} gs://anrl-storage/results/'.format(file_name))

# deepFogGuard Plus Ablation experiment
def dropout_ablation():
    # get cifar10 data 
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
     # normalize input
    x_train = x_train / 255
    x_test = x_test / 255
    # Concatenate train and test images
    x = np.concatenate((x_train,x_test))
    y = np.concatenate((y_train,y_test))

    # split data into train, validation, and holdout set (80/10/10)
    x_train, x_test, y_train, y_test = train_test_split(x,y,random_state = 42, test_size = .20, shuffle = True)
    x_val, x_test, y_val, y_test = train_test_split(x_test,y_test,random_state = 42, test_size = .50, shuffle = True)
    train_datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    )
    survive_configs = [
        [.96,.98],
        [.90,.95],
        [.80,.85],
        [1,1]
    ]
    num_iterations = 10
    output = {
        "DeepFogGuard Plus Baseline":
        {
            "[0.9, 0.9, 0.9]":
            {
                "[0.96, 0.98]": [0] * num_iterations,
                "[0.9, 0.95]":[0] * num_iterations,
                "[0.8, 0.85]":[0] * num_iterations,
                "[1, 1]":[0] * num_iterations,
            },
            "[0.7, 0.7, 0.7]":
            {
               "[0.96, 0.98]": [0] * num_iterations,
                "[0.9, 0.95]":[0] * num_iterations,
                "[0.8, 0.85]":[0] * num_iterations,
                "[1, 1]":[0] * num_iterations,
            },
            "[0.5, 0.5, 0.5]":
            {
                "[0.96, 0.98]": [0] * num_iterations,
                "[0.9, 0.95]":[0] * num_iterations,
                "[0.8, 0.85]":[0] * num_iterations,
                "[1, 1]":[0] * num_iterations,
            },
            "[0.95, 0.95, 0.95]":
            {
                "[0.96, 0.98]": [0] * num_iterations,
                "[0.9, 0.95]":[0] * num_iterations,
                "[0.8, 0.85]":[0] * num_iterations,
                "[1, 1]":[0] * num_iterations,
            },
        }
    }
    dropout_configs = [
        [.9,.9,.9],
        [.7,.7,.7],
        [.5,.5,.5],
        [.95,.95,.95],
    ]
    now = datetime.datetime.now()
    date = str(now.month) + '-' + str(now.day) + '-' + str(now.year)
    # make folder for outputs 
    if not os.path.exists('results/' + date):
        os.mkdir('results/')
        os.mkdir('results/' + date)
    file_name = 'results/' + date + '/experiment3_dropoutAblation_fixedsplit_results.txt'
    output_list = []
    batch_size = 128
    train_steps_per_epoch = math.ceil(len(x_train) / batch_size)
    val_steps_per_epoch = math.ceil(len(x_val) / batch_size)
    for iteration in range(1,num_iterations+1):
        print("iteration:",iteration)
        for dropout in dropout_configs:
            model_name = "GitHubANRL_deepFogGuardPlus_dropoutAblation" + str(dropout) + "_" + str(iteration) + ".h5"
            model = define_deepFogGuardPlus_CNN(weights = None,classes=10,input_shape = (32,32,3),dropout = 0, alpha = .5,survive_rates=dropout)
            modelCheckPoint = ModelCheckpoint(model_name, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
            model.fit_generator(train_datagen.flow(x_train,y_train,batch_size = batch_size),
            epochs = 75,
            validation_data = (x_val,y_val), 
            steps_per_epoch = train_steps_per_epoch, 
            verbose = 2, 
            validation_steps = val_steps_per_epoch,
            callbacks = [modelCheckPoint])
            # load weights with the highest validaton acc
            model.load_weights(model_name)
            for survive_config in survive_configs:
                output_list.append(str(survive_config) + '\n')
                print(survive_config)
                output["DeepFogGuard Plus Baseline"][str(dropout)][str(survive_config)][iteration-1] = run(model, survive_config,output_list, y_train, x_test, y_test)
            # clear session so that model will recycled back into memory
            K.clear_session()
            gc.collect()
            del model
    with open(file_name,'a+') as file:
        for survive_config in survive_configs:
            for dropout in dropout_configs:
                output_list.append(str(survive_config) + '\n')
                deepGuardPlus_acc = average(output["DeepFogGuard Plus Baseline"][str(dropout)][str(survive_config)])
                output_list.append(str(survive_config) + str(dropout) + " Dropout Accuracy: " + str(deepGuardPlus_acc) + '\n')
                print(str(survive_config), str(dropout), " Dropout Accuracy:",deepGuardPlus_acc)
        file.writelines(output_list)
        file.flush()
        os.fsync(file)
    use_GCP = True
    if use_GCP:
        os.system('gsutil -m -q cp -r *.h5 gs://anrl-storage/models')
        os.system('gsutil -m -q cp -r {} gs://anrl-storage/results/'.format(file_name))

# deepFogGuard hyperconnection weight ablation experiment      
def hyperconnection_weight_ablation():
     # get cifar10 data 
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # normalize input
    x_train = x_train / 255
    x_test = x_test / 255
    # Concatenate train and test images
    x = np.concatenate((x_train,x_test))
    y = np.concatenate((y_train,y_test))

    # split data into train, validation, and holdout set (80/10/10)
    x_train, x_test, y_train, y_test = train_test_split(x,y,random_state = 42, test_size = .20, shuffle = True)
    x_val, x_test, y_val, y_test = train_test_split(x_test,y_test,random_state = 42, test_size = .50, shuffle = True)
    train_datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    )
    survive_configs = [
        [.96,.98],
        [.90,.95],
        [.80,.85],
        [1,1]
    ]
    num_iterations = 10
    output = {
        "DeepFogGuard Baseline":
        {
            "[0.96, 0.98]": [0] * num_iterations,
            "[0.9, 0.95]":[0] * num_iterations,
            "[0.8, 0.85]":[0] * num_iterations,
            "[1, 1]":[0] * num_iterations,
        },
    }
    now = datetime.datetime.now()
    date = str(now.month) + '-' + str(now.day) + '-' + str(now.year)
    # make folder for outputs 
    if not os.path.exists('results/' + date):
        os.mkdir('results/')
        os.mkdir('results/' + date)
    file_name = 'results/' + date + '/experiment3_hyperconnection_weight_ablation_weightedbys(i)_fixedsplit_results.txt'
    output_list = []
    for iteration in range(1,num_iterations+1):
        print("iteration:",iteration)
        batch_size = 128
        train_steps_per_epoch = math.ceil(len(x_train) / batch_size)
        val_steps_per_epoch = math.ceil(len(x_val) / batch_size)
        for survive_config in survive_configs:
            model_name = "GitHubANRL_deepFogGuard_hyperconnectionweightablation_weightedbys(i)_" + str(survive_config) + "_fixedsplit_" + str(iteration) + ".h5"
            model = define_deepFogGuard_CNN(weights = None,classes=10,input_shape = (32,32,3),dropout = 0, alpha = .5,hyperconnection_weights=survive_config, hyperconnection_weights_scheme = 2)
            modelCheckPoint = ModelCheckpoint(model_name, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
            model.fit_generator(train_datagen.flow(x_train,y_train,batch_size = batch_size),
            epochs = 75,
            validation_data = (x_val,y_val), 
            steps_per_epoch = train_steps_per_epoch, 
            verbose = 2, 
            validation_steps = val_steps_per_epoch,
            callbacks = [modelCheckPoint])
            # load weights with the highest validaton acc
            model.load_weights(model_name)
            output_list.append(str(survive_config) + '\n')
            print(survive_config)
            output["DeepFogGuard Baseline"][str(survive_config)][iteration-1] = run(model, survive_config,output_list, y_train, x_test, y_test)
            # clear session so that model will recycled back into memory
            K.clear_session()
            gc.collect()
            del model
    with open(file_name,'a+') as file:
        for survive_config in survive_configs:
            output_list.append(str(survive_config) + '\n')
            active_guard_acc = average(output["DeepFogGuard Baseline"][str(survive_config)])
            output_list.append(str(survive_config) + str(active_guard_acc) + '\n')
            print(str(survive_config),active_guard_acc)
        file.writelines(output_list)
        file.flush()
        os.fsync(file)
    use_GCP = True
    if use_GCP:
        os.system('gsutil -m -q cp -r *.h5 gs://anrl-storage/models')
        os.system('gsutil -m -q cp -r {} gs://anrl-storage/results/'.format(file_name))

# deepFogGuard hyperconnection failure configuration ablation experiment
def hyperconnection_sensitivity_ablation():
    # get cifar10 data 
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # normalize input
    x_train = x_train / 255
    x_test = x_test / 255
    # Concatenate train and test images
    x = np.concatenate((x_train,x_test))
    y = np.concatenate((y_train,y_test))

    # split data into train, validation, and holdout set (80/10/10)
    x_train, x_test, y_train, y_test = train_test_split(x,y,random_state = 42, test_size = .20, shuffle = True)
    x_val, x_test, y_val, y_test = train_test_split(x_test,y_test,random_state = 42, test_size = .50, shuffle = True)
    train_datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    )
    survive_configs = [
        [.96,.98],
        [.90,.95],
        [.80,.85],
    ]
    num_iterations = 20
    hyperconnections = [
        [0,0],
        [1,0],
        [0,1],
        [1,1],
    ]
    output = {
        "DeepFogGuard Hyperconnection Sensitivity":
        {
            "[0.96, 0.98]":      
            {  
                "[0, 0]":[0] * num_iterations,
                "[1, 0]":[0] * num_iterations,
                "[0, 1]":[0] * num_iterations,
                "[1, 1]":[0] * num_iterations,
            },
            "[0.9, 0.95]":
            {
                "[0, 0]":[0] * num_iterations,
                "[1, 0]":[0] * num_iterations,
                "[0, 1]":[0] * num_iterations,
                "[1, 1]":[0] * num_iterations,
            },
            "[0.8, 0.85]":
            {
                "[0, 0]":[0] * num_iterations,
                "[1, 0]":[0] * num_iterations,
                "[0, 1]":[0] * num_iterations,
                "[1, 1]":[0] * num_iterations,
            },
        },
    }
    now = datetime.datetime.now()
    date = str(now.month) + '-' + str(now.day) + '-' + str(now.year)
    # make folder for outputs 
    if not os.path.exists('results/' + date):
        os.mkdir('results/')
        os.mkdir('results/' + date)
    file_name = 'results/' + date + '/experiment3_hyperconnection_sensitivityablation_fixedsplit_results3.txt'
    output_list = []
    for iteration in range(1,num_iterations+1):
        print("iteration:",iteration)
        batch_size = 128
        train_steps_per_epoch = math.ceil(len(x_train) / batch_size)
        val_steps_per_epoch = math.ceil(len(x_val) / batch_size)
        for hyperconnection in hyperconnections:
            model_name = "GitHubANRL_deepFogGuardPlus_hyperconnectionsensitvityablation" + str(hyperconnection) + "_fixedsplit_" + str(iteration) + ".h5"
            model = define_deepFogGuard_CNN(weights = None,classes=10,input_shape = (32,32,3),dropout = 0, alpha = .5,hyperconnections = hyperconnection)
            modelCheckPoint = ModelCheckpoint(model_name, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
            model.fit_generator(train_datagen.flow(x_train,y_train,batch_size = batch_size),
            epochs = 75,
            validation_data = (x_val,y_val), 
            steps_per_epoch = train_steps_per_epoch, 
            verbose = 2, 
            validation_steps = val_steps_per_epoch,
            callbacks = [modelCheckPoint])
            # load weights with the highest validaton acc
            model.load_weights(model_name)
            for survive_config in survive_configs:
                output_list.append(str(survive_config) + '\n')
                print(survive_config)
                output["DeepFogGuard Hyperconnection Sensitivity"][str(survive_config)][str(hyperconnection)][iteration-1] = run(model, survive_config,output_list, y_train, x_test, y_test)
            # clear session so that model will recycled back into memory
            K.clear_session()
            gc.collect()
            del model
    with open(file_name,'a+') as file:
        for survive_config in survive_configs:
            for hyperconnection in hyperconnections:
                output_list.append(str(survive_config) + '\n')
                active_guard_acc = average(output["DeepFogGuard Hyperconnection Sensitivity"][str(survive_config)][str(hyperconnection)])
                acc_std = np.std(output["DeepFogGuard Hyperconnection Sensitivity"][str(survive_config)][str(hyperconnection)],ddof=1)
                output_list.append(str(survive_config) + str(hyperconnection) + str(active_guard_acc) + '\n')
                output_list.append(str(survive_config) + str(hyperconnection) + str(acc_std) + '\n')
                print(str(survive_config),active_guard_acc)
                print(str(survive_config), acc_std)
        file.writelines(output_list)
        file.flush()
        os.fsync(file)
    use_GCP = True
    if use_GCP:
        os.system('gsutil -m -q cp -r *.h5 gs://anrl-storage/models')
        os.system('gsutil -m -q cp -r {} gs://anrl-storage/results/'.format(file_name))

# cnn experiment 
if __name__ == "__main__":
    #cnn_normal_experiments()
    #dropout_ablation()
    #hyperconnection_weight_ablation()
    hyperconnection_sensitivity_ablation()