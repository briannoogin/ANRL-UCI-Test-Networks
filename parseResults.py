import statistics
import pprint
def parseReport(fileName):
    acc_list = []
    with open(fileName,'r') as file:
        for line in file:
            index = line.find("Average Accuracy: ")
            if index != -1:
                split_line = line.split()
                acc = float(split_line[-1])
                acc_list.append(acc)
    survive_configurations = [
        [.78,.8,.85],
        [.87,.91,.95],
        [.92,.96,.99]
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
    report_index = 0
    start_iteration = 16
    end_iteration = 20
    for iteration in range(start_iteration,end_iteration + 1):
        print("Trial",iteration)
        for survive_configuration in survive_configurations:
            print(survive_configuration)
            for hyperconnection in hyperconnections:
                print(hyperconnection)
                print(survive_configuration,hyperconnection,"ActiveGuard Accuracy:",acc_list[report_index])
                print(survive_configuration,hyperconnection,"FixedGuard Accuracy:",acc_list[report_index + 1])
                report_index+=2

def calculateStatsReportName(reportName):
    num_iterations = 20
    survive_configurations = [
        [.78,.8,.85],
        [.87,.91,.95],
        [.92,.96,.99]
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
    report = {
        "Active Guard":
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
        }, 
        "Fixed Guard":
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
        }
    }
    stats_report = {
        "Active Guard":
        {
            "[0.78, 0.8, 0.85]":
            {
                "[0, 0, 0]":
                {
                    "mean":0,
                    "std":0
                },
                "[1, 0, 0]":   
                {
                    "mean":0,
                    "std":0
                },
                "[0, 1, 0]":   
                {
                    "mean":0,
                    "std":0
                },
                "[0, 0, 1]":   
                {
                    "mean":0,
                    "std":0
                },
                "[1, 1, 0]":   
                {
                    "mean":0,
                    "std":0
                },
                "[1, 0, 1]":   
                {
                    "mean":0,
                    "std":0
                },
                "[0, 1, 1]":   
                {
                    "mean":0,
                    "std":0
                },
                "[1, 1, 1]":   
                {
                    "mean":0,
                    "std":0
                }
            },
            "[0.87, 0.91, 0.95]":
            {
                "[0, 0, 0]":
                {
                    "mean":0,
                    "std":0
                },
                "[1, 0, 0]":   
                {
                    "mean":0,
                    "std":0
                },
                "[0, 1, 0]":   
                {
                    "mean":0,
                    "std":0
                },
                "[0, 0, 1]":   
                {
                    "mean":0,
                    "std":0
                },
                "[1, 1, 0]":   
                {
                    "mean":0,
                    "std":0
                },
                "[1, 0, 1]":   
                {
                    "mean":0,
                    "std":0
                },
                "[0, 1, 1]":   
                {
                    "mean":0,
                    "std":0
                },
                "[1, 1, 1]":   
                {
                    "mean":0,
                    "std":0
                }
            },
            "[0.92, 0.96, 0.99]":
            {
             "[0, 0, 0]":
                {
                    "mean":0,
                    "std":0
                },
                "[1, 0, 0]":   
                {
                    "mean":0,
                    "std":0
                },
                "[0, 1, 0]":   
                {
                    "mean":0,
                    "std":0
                },
                "[0, 0, 1]":   
                {
                    "mean":0,
                    "std":0
                },
                "[1, 1, 0]":   
                {
                    "mean":0,
                    "std":0
                },
                "[1, 0, 1]":   
                {
                    "mean":0,
                    "std":0
                },
                "[0, 1, 1]":   
                {
                    "mean":0,
                    "std":0
                },
                "[1, 1, 1]":   
                {
                    "mean":0,
                    "std":0
                }
            },
        }, 
        "Fixed Guard":
        {
            "[0.78, 0.8, 0.85]":
            {
                "[0, 0, 0]":
                {
                    "mean":0,
                    "std":0
                },
                "[1, 0, 0]":   
                {
                    "mean":0,
                    "std":0
                },
                "[0, 1, 0]":   
                {
                    "mean":0,
                    "std":0
                },
                "[0, 0, 1]":   
                {
                    "mean":0,
                    "std":0
                },
                "[1, 1, 0]":   
                {
                    "mean":0,
                    "std":0
                },
                "[1, 0, 1]":   
                {
                    "mean":0,
                    "std":0
                },
                "[0, 1, 1]":   
                {
                    "mean":0,
                    "std":0
                },
                "[1, 1, 1]":   
                {
                    "mean":0,
                    "std":0
                }
            },
            "[0.87, 0.91, 0.95]":
            {
                "[0, 0, 0]":
                {
                    "mean":0,
                    "std":0
                },
                "[1, 0, 0]":   
                {
                    "mean":0,
                    "std":0
                },
                "[0, 1, 0]":   
                {
                    "mean":0,
                    "std":0
                },
                "[0, 0, 1]":   
                {
                    "mean":0,
                    "std":0
                },
                "[1, 1, 0]":   
                {
                    "mean":0,
                    "std":0
                },
                "[1, 0, 1]":   
                {
                    "mean":0,
                    "std":0
                },
                "[0, 1, 1]":   
                {
                    "mean":0,
                    "std":0
                },
                "[1, 1, 1]":   
                {
                    "mean":0,
                    "std":0
                }
            },
            "[0.92, 0.96, 0.99]":
            {
             "[0, 0, 0]":
                {
                    "mean":0,
                    "std":0
                },
                "[1, 0, 0]":   
                {
                    "mean":0,
                    "std":0
                },
                "[0, 1, 0]":   
                {
                    "mean":0,
                    "std":0
                },
                "[0, 0, 1]":   
                {
                    "mean":0,
                    "std":0
                },
                "[1, 1, 0]":   
                {
                    "mean":0,
                    "std":0
                },
                "[1, 0, 1]":   
                {
                    "mean":0,
                    "std":0
                },
                "[0, 1, 1]":   
                {
                    "mean":0,
                    "std":0
                },
                "[1, 1, 1]":   
                {
                    "mean":0,
                    "std":0
                }
            }
        }
    }
    acc_list = []
    report_index = 0
    with open(reportName,'r') as file:
        for line in file:
            index = line.find("Accuracy: ")
            if index != -1:
                split_line = line.split()
                acc = float(split_line[-1])
                acc_list.append(acc)
    trials = 20
    # enter all the data into the dictionary 
    for iteration in range(0,trials):
        for survive_configuration in survive_configurations:
            for hyperconnection in hyperconnections:
                report["Active Guard"][str(survive_configuration)][str(hyperconnection)][iteration] = acc_list[report_index]
                report["Fixed Guard"][str(survive_configuration)][str(hyperconnection)][iteration] = acc_list[report_index + 1]
                report_index+=2
    # calculate mean and standard deviation
    for survive_configuration in survive_configurations:
        for hyperconnection in hyperconnections:
            # calculate mean
            stats_report["Active Guard"][str(survive_configuration)][str(hyperconnection)]["mean"] = statistics.mean(report["Active Guard"][str(survive_configuration)][str(hyperconnection)])
            stats_report["Fixed Guard"][str(survive_configuration)][str(hyperconnection)]["mean"] = statistics.mean(report["Fixed Guard"][str(survive_configuration)][str(hyperconnection)])
            # calculate std
            stats_report["Active Guard"][str(survive_configuration)][str(hyperconnection)]["std"] = statistics.stdev(report["Active Guard"][str(survive_configuration)][str(hyperconnection)])
            stats_report["Fixed Guard"][str(survive_configuration)][str(hyperconnection)]["std"] = statistics.stdev(report["Fixed Guard"][str(survive_configuration)][str(hyperconnection)])
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(stats_report)
if __name__ == "__main__":
    #parseReport('results16_20.txt')
    calculateStatsReportName('ANRL Sensitivity Analysis Results.txt')