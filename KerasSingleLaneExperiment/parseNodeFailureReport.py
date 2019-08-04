import numpy as np
def parse(file_name):
  
    report_dict = {
        "deepFogGuard Plus":
        {
            "[0, 0, 0]": [0] * 10,
            "[0, 0, 1]": [0] * 10,
            "[0, 1, 0]": [0] * 10,
            "[0, 1, 1]": [0] * 10,
            "[1, 0, 0]": [0] * 10,
            "[1, 0, 1]": [0] * 10,
            "[1, 1, 0]": [0] * 10,
            "[1, 1, 1]": [0] * 10,
        },
        "deepFogGuard":
        {
            "[0, 0, 0]": [0] * 10,
            "[0, 0, 1]": [0] * 10,
            "[0, 1, 0]": [0] * 10,
            "[0, 1, 1]": [0] * 10,
            "[1, 0, 0]": [0] * 10,
            "[1, 0, 1]": [0] * 10,
            "[1, 1, 0]": [0] * 10,
            "[1, 1, 1]": [0] * 10,
        }
        ,
        "Vanilla":
        {
            "[0, 0, 0]": [0] * 10,
            "[0, 0, 1]": [0] * 10,
            "[0, 1, 0]": [0] * 10,
            "[0, 1, 1]": [0] * 10,
            "[1, 0, 0]": [0] * 10,
            "[1, 0, 1]": [0] * 10,
            "[1, 1, 0]": [0] * 10,
            "[1, 1, 1]": [0] * 10,
        }
    }
    avg_dict = {
        "deepFogGuard Plus":
        {
            "[0, 0, 0]": 0,
            "[0, 0, 1]": 0,
            "[0, 1, 0]": 0,
            "[0, 1, 1]": 0,
            "[1, 0, 0]": 0,
            "[1, 0, 1]": 0,
            "[1, 1, 0]": 0,
            "[1, 1, 1]": 0,
        },
        "deepFogGuard":
        {
            "[0, 0, 0]": 0,
            "[0, 0, 1]": 0,
            "[0, 1, 0]": 0,
            "[0, 1, 1]": 0,
            "[1, 0, 0]": 0,
            "[1, 0, 1]": 0,
            "[1, 1, 0]": 0,
            "[1, 1, 1]": 0,
        }
        ,
        "Vanilla":
        {
            "[0, 0, 0]": 0,
            "[0, 0, 1]": 0,
            "[0, 1, 0]": 0,
            "[0, 1, 1]": 0,
            "[1, 0, 0]": 0,
            "[1, 0, 1]": 0,
            "[1, 1, 0]": 0,
            "[1, 1, 1]": 0,
        }
    }
    num_iterations = 1
    model_counter = 0
    counter = 0
    # goes from low survival config to high 
    with open(file_name) as file:

        for line in file:
            index = line.find("acc:")
            if index != -1:
                split_line = line.split()
                acc = float(split_line[-1])
                # extract the survival config from the file
                survival_config = line.split('n')[0]
                if model_counter % 3 == 0:
                    report_dict['deepFogGuard Plus'][survival_config][num_iterations-1] = acc
                    print(acc)
                elif model_counter % 3 == 1:
                    report_dict['deepFogGuard'][survival_config][num_iterations-1] = acc
                else:
                    report_dict['Vanilla'][survival_config][num_iterations-1] = acc
                counter+=1 
                if(counter % 96 == 0):
                    num_iterations+=1
                if(counter % 8 == 0):
                    model_counter+=1

    for model in report_dict:
        for survival_config in report_dict[model]:
            avg_dict[model][survival_config] = np.average(report_dict[model][survival_config])
    print(avg_dict)
if __name__ == "__main__":
    parse("results_newsplit_normalHealthActivityExperiment.txt")