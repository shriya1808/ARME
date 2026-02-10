# Import necessary libraries for  association rule mining
import copy
import time
import numpy as np
import pandas as pd
#from mlxtend.frequent_patterns import apriori, fpmax, fpgrowth
#from mlxtend.frequent_patterns import association_rules
from ARMAE import *
import torch
import os
import argparse

parser = argparse.ArgumentParser(description="MIMIC-IV Data Pipeline")
parser.add_argument(
    "--workdir",
    type=str,
    required=True,
    help="Working directory for the pipeline"
)
args = parser.parse_args()

root_dir = os.path.dirname(args.workdir)
print(os.getcwd())

# Setting parameters for the algorithm and training
nbAntecedents_algo = 3  # Number of antecedents for the algorithm
minSupp_algo = 0.01  # Minimum support for the algorithm
minConf = 0.1  # Minimum confidence for the algorithm
nbRun = 2  # Number of runs
nbEpoch = 2  # Number of epochs for training
batchSize = 256  # Batch size for training
learningRate = 10e-3  # Learning rate for training
likeness = 0.5  # Proportion of similar items in rule with the same consequents
numberOfRules = 1  # Number of rules per consequent
minSupp_AE = 0.01  # Minimum support for FP-Growth

# Define file paths and dataset details
datasetName = 'patient_features_COPD_cleaned'

ARMAEResultsPath = os.path.join(root_dir, 'Results', 'NN') 
exhaustiveResultsPath = os.path.join(root_dir, 'Results', 'Exhaustive')
overallResultsPath = os.path.join(root_dir, 'Results', 'Final')
dataPath = os.path.join(root_dir, 'data', 'clusters', datasetName + '.csv')

isLoadedModel = False  # Flag to check if model is loaded

# Paths to save the models
modelPath = os.path.join(root_dir, 'models/')
encoderPath = os.path.join(modelPath, '1encoder.pt')  # Pretrained encoder path
decoderPath = os.path.join(modelPath, '1decoder.pt')  # Pretrained decoder path

# Load the dataset
data = pd.read_csv(dataPath)
#data.drop(columns=['label_0'], inplace=True)  # Drop the 'label_0' column from the dataset
times = []

dataSize = len(data.loc[0])  # Get the size of the data
nbAntecedents_AE = dataSize  # Set the number of antecedents for AE

print("size of the data is", dataSize)

'''
# Function to take user input for choosing the algorithm
def Input():
    print("Choose which model you want to run: (a) apriori, (b) fpgrowth: ")
    user_input = input("Enter your choice: ")
    output = str
    if user_input in ["a", "A"]:
        output = "Apriori"
    elif user_input in ["b", "B"]:
        output = "FPGrowth"
    else:
        print("Invalid entry. Please enter 'a', 'b'.")
    return output

choice = Input()  # Get user's choice of algorithm
print(f"......{choice} Rule Mining Starts here.............")

# Function to perform exhaustive search using Apriori or FP-Growth algorithms
def ExhaustiveSearch(df, path_laws, minSupp_algo, min_conf, user_input, nbAntecedent=1, dataset='Mushroom'):
    # Select the appropriate algorithm based on user input
    if user_input == "Apriori":
        frequent_itemsets = apriori(df, min_support=minSupp_algo, use_colnames=True, max_len=nbAntecedent+1)
        al_name = "Apriori"
        print('Apriori Itemsets')
    elif user_input == "FPGrowth":
        frequent_itemsets = fpgrowth(df, min_support=minSupp_algo, use_colnames=True, max_len=nbAntecedent+1)
        al_name = "FPGrowth"
        print('FPGrowth Itemsets')
    else:
        print("Invalid entry")
    print(frequent_itemsets)

    # Generate association rules from the frequent itemsets
    exhauRules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_conf)
    print(exhauRules)
    nbLoisInit = str(len(exhauRules))

    # Define exclusion list for antecedents
    exclusion_list = [
        "re.admission.within.28.days_yes", "re.admission.within.28.days_no",
        "re.admission.btw.28days.to.3months_yes", "re.admission.btw.28days.to.3months_no",
        "re.admission.btw.3months.to.6months_yes", "re.admission.btw.3months.to.6months_no"
    ]

    # Filter rules based on exclusion list and consequents criteria
    exhauRules = exhauRules[~exhauRules['antecedents'].apply(lambda x: any(item in exclusion_list for item in x))]
    exhauRules = exhauRules[exhauRules['consequents'].apply(lambda x: len(x) == 1 and (
        "re.admission.within.28.days_yes" in x or "re.admission.within.28.days_no" in x or
        "re.admission.btw.28days.to.3months_yes" in x or "re.admission.btw.28days.to.3months_no" in x or
        "re.admission.btw.3months.to.6months_yes" in x or "re.admission.btw.3months.to.6months_no" in x
    ))]
    exhauRules = exhauRules.reset_index()
    exhauRules = exhauRules.sort_values(by=['support'], ignore_index=True, ascending=False)

    # Separate rules based on different time frames for readmission
    consequents_1_yes = ["re.admission.within.28.days_yes"]
    consequents_1_no = ["re.admission.within.28.days_no"]
    readmission_28_days_yes = exhauRules[exhauRules['consequents'].apply(lambda x: len(x) == 1 and any(item in x for item in consequents_1_yes))]
    readmission_28_days_no = exhauRules[exhauRules['consequents'].apply(lambda x: len(x) == 1 and any(item in x for item in consequents_1_no))]

    consequents_2_yes = ["re.admission.btw.28days.to.3months_yes"]
    consequents_2_no = ["re.admission.btw.28days.to.3months_no"]
    readmission_3_months_yes = exhauRules[exhauRules['consequents'].apply(lambda x: len(x) == 1 and any(item in x for item in consequents_2_yes))]
    readmission_3_months_no = exhauRules[exhauRules['consequents'].apply(lambda x: len(x) == 1 and any(item in x for item in consequents_2_no))]

    consequents_3_yes = ["re.admission.btw.3months.to.6months_yes"]
    consequents_3_no = ["re.admission.btw.3months.to.6months_no"]
    readmission_6_months_yes = exhauRules[exhauRules['consequents'].apply(lambda x: len(x) == 1 and any(item in x for item in consequents_3_yes))]
    readmission_6_months_no = exhauRules[exhauRules['consequents'].apply(lambda x: len(x) == 1 and any(item in x for item in consequents_3_no))]

    # Check Rule Quality Of Readmission
    good_quality_rules = exhauRules[(exhauRules['lift'] > 1.2) &
                                    (exhauRules['support'] > 0.01) &
                                    (exhauRules['confidence'] > 0.6)]
    good_quality_rules_28_days_yes = good_quality_rules[good_quality_rules['consequents'].apply(lambda x: len(x) == 1 and any(item in x for item in consequents_1_yes))]
    good_quality_rules_28_days_no = good_quality_rules[good_quality_rules['consequents'].apply(lambda x: len(x) == 1 and any(item in x for item in consequents_1_no))]
    good_quality_rules_3_months_yes = good_quality_rules[good_quality_rules['consequents'].apply(lambda x: len(x) == 1 and any(item in x for item in consequents_2_yes))]
    good_quality_rules_3_months_no = good_quality_rules[good_quality_rules['consequents'].apply(lambda x: len(x) == 1 and any(item in x for item in consequents_2_no))]
    good_quality_rules_6_months_yes = good_quality_rules[good_quality_rules['consequents'].apply(lambda x: len(x) == 1 and any(item in x for item in consequents_3_yes))]
    good_quality_rules_6_months_no = good_quality_rules[good_quality_rules['consequents'].apply(lambda x: len(x) == 1 and any(item in x for item in consequents_3_no))]

    bad_quality_rules = exhauRules[exhauRules['lift'] < 1]
    bad_quality_rules_28_days_yes = bad_quality_rules[bad_quality_rules['consequents'].apply(lambda x: len(x) == 1 and any(item in x for item in consequents_1_yes))]
    bad_quality_rules_28_days_no = bad_quality_rules[bad_quality_rules['consequents'].apply(lambda x: len(x) == 1 and any(item in x for item in consequents_1_no))]
    bad_quality_rules_3_months_yes = bad_quality_rules[bad_quality_rules['consequents'].apply(lambda x: len(x) == 1 and any(item in x for item in consequents_2_yes))]
    bad_quality_rules_3_months_no = bad_quality_rules[bad_quality_rules['consequents'].apply(lambda x: len(x) == 1 and any(item in x for item in consequents_2_no))]
    bad_quality_rules_6_months_yes = bad_quality_rules[bad_quality_rules['consequents'].apply(lambda x: len(x) == 1 and any(item in x for item in consequents_3_yes))]
    bad_quality_rules_6_months_no = bad_quality_rules[bad_quality_rules['consequents'].apply(lambda x: len(x) == 1 and any(item in x for item in consequents_3_no))]

    # Write good quality rules to CSV
    path_good_rules = "Results/Final/readmission_cluster_1_apriori_good_quality_rules.csv"
    good_quality_rules.to_csv(path_good_rules, index=False)

    # Calculate statistics for the rules
    good_quality_rules_count = len(good_quality_rules)
    bad_quality_rules_count = len(bad_quality_rules)
    total_count = len(exhauRules)
    good_quality_rules_percentage = (good_quality_rules_count / total_count) * 100 if total_count != 0 else 0
    bad_quality_rules_percentage = (bad_quality_rules_count / total_count) * 100 if total_count != 0 else 0

    rule_length = len(exhauRules)
    supportAvg = np.mean(exhauRules['support'])
    confAvg = np.mean(exhauRules['confidence'])

    # Calculate average support and confidence for each category
    avg_support_28_days_yes = np.mean(readmission_28_days_yes['support']) if len(readmission_28_days_yes) > 0 else 0
    avg_confidence_28_days_yes = np.mean(readmission_28_days_yes['confidence']) if len(readmission_28_days_yes) > 0 else 0
    avg_support_28_days_no = np.mean(readmission_28_days_no['support']) if len(readmission_28_days_no) > 0 else 0
    avg_confidence_28_days_no = np.mean(readmission_28_days_no['confidence']) if len(readmission_28_days_no) > 0 else 0
    avg_support_3_months_yes = np.mean(readmission_3_months_yes['support']) if len(readmission_3_months_yes) > 0 else 0
    avg_confidence_3_months_yes = np.mean(readmission_3_months_yes['confidence']) if len(readmission_3_months_yes) > 0 else 0
    avg_support_3_months_no = np.mean(readmission_3_months_no['support']) if len(readmission_3_months_no) > 0 else 0
    avg_confidence_3_months_no = np.mean(readmission_3_months_no['confidence']) if len(readmission_3_months_no) > 0 else 0
    avg_support_6_months_yes = np.mean(readmission_6_months_yes['support']) if len(readmission_6_months_yes) > 0 else 0
    avg_confidence_6_months_yes = np.mean(readmission_6_months_yes['confidence']) if len(readmission_6_months_yes) > 0 else 0
    avg_support_6_months_no = np.mean(readmission_6_months_no['support']) if len(readmission_6_months_no) > 0 else 0
    avg_confidence_6_months_no = np.mean(readmission_6_months_no['confidence']) if len(readmission_6_months_no) > 0 else 0

    exhauRules.to_csv(path_laws, index=False)
    good_quality_rules['antecedents'] = good_quality_rules['antecedents'].apply(lambda x: tuple(x))
    good_quality_rules['consequents'] = good_quality_rules['consequents'].apply(lambda x: tuple(x))
    good_quality_rules.drop(["index"], axis=1, inplace=True)
    path_good_rules = "Results/Final/readmission_cluster_1_apriori_good_quality_rules.csv"

    good_quality_rules.to_csv(path_good_rules, index=False)

    return [supportAvg, confAvg, rule_length, good_quality_rules, bad_quality_rules, al_name, nbLoisInit,
            good_quality_rules_percentage, bad_quality_rules_percentage, readmission_28_days_yes, readmission_28_days_no,
            readmission_3_months_yes, readmission_3_months_no, readmission_6_months_yes, readmission_6_months_no,
            good_quality_rules_28_days_yes, good_quality_rules_28_days_no, good_quality_rules_3_months_yes, good_quality_rules_3_months_no,
            good_quality_rules_6_months_yes, good_quality_rules_6_months_no,
            avg_support_28_days_yes, avg_confidence_28_days_yes, avg_support_28_days_no, avg_confidence_28_days_no,
            avg_support_3_months_yes, avg_confidence_3_months_yes, avg_support_3_months_no, avg_confidence_3_months_no,
            avg_support_6_months_yes, avg_confidence_6_months_yes, avg_support_6_months_no, avg_confidence_6_months_no]

# Execute exhaustive search based on user choice
if choice in ["Apriori", "FPGrowth"]:
    t1 = time.time()  # Start timing the execution
    firstScore = ExhaustiveSearch(data, overallResultsPath + datasetName + '.csv', minSupp_algo, minConf, choice,
                                  nbAntecedent=nbAntecedents_algo, dataset=datasetName)
    t2 = time.time()  # End timing the execution
    total_time = t2 - t1  # Calculate total execution time
    print("Total Number of Rules : ", firstScore[6])
    print("Total Number of Readmission Rules : ", firstScore[2])
    print("Support Average is  : ", firstScore[0])
    print("Confidence Average is  : ", firstScore[1])
    print("Quality of Rules is Evaluated on the basis of lift. If lift is greater than 1 rules are classified as Good Quality Rules")
    print("Total Number of Good Quality Rules : ", len(firstScore[3]))
    print("Total Number of Readmission within 28 Days (Yes) Rules : ", len(firstScore[9]))
    print("Total Number of Readmission within 28 Days (No) Rules : ", len(firstScore[10]))
    print("Total Number of Readmission between 28 Days to 3 Months (Yes) Rules : ", len(firstScore[11]))
    print("Total Number of Readmission between 28 Days to 3 Months (No) Rules : ", len(firstScore[12]))
    print("Total Number of Readmission between 3 Months to 6 Months (Yes) Rules : ", len(firstScore[13]))
    print("Total Number of Readmission between 3 Months to 6 Months (No) Rules : ", len(firstScore[14]))
    print("Good Quality of Rules Percentage : ", firstScore[7])
    print("Bad Quality of Rules Percentage : ", firstScore[8])
    print("Total Number of Good Readmission within 28 Days (Yes) Rules : ", len(firstScore[15]))
    print("Total Number of Good Readmission within 28 Days (No) Rules : ", len(firstScore[16]))
    print("Total Number of Good Readmission between 28 Days to 3 Months (Yes) Rules : ", len(firstScore[17]))
    print("Total Number of Good Readmission between 28 Days to 3 Months (No) Rules : ", len(firstScore[18]))
    print("Total Number of Good Readmission between 3 Months to 6 Months (Yes) Rules : ", len(firstScore[19]))
    print("Total Number of Good Readmission between 3 Months to 6 Months (No) Rules : ", len(firstScore[20]))

    # Calculate and print percentages for good quality rules
    per_good_quality_28_days_yes = np.round((len(firstScore[15]) / len(firstScore[9])) * 100, 2) if len(firstScore[9]) != 0 else 0
    per_good_quality_28_days_no = np.round((len(firstScore[16]) / len(firstScore[10])) * 100, 2) if len(firstScore[10]) != 0 else 0
    per_good_quality_3_months_yes = np.round((len(firstScore[17]) / len(firstScore[11])) * 100, 2) if len(firstScore[11]) != 0 else 0
    per_good_quality_3_months_no = np.round((len(firstScore[18]) / len(firstScore[12])) * 100, 2) if len(firstScore[12]) != 0 else 0
    per_good_quality_6_months_yes = np.round((len(firstScore[19]) / len(firstScore[13])) * 100, 2) if len(firstScore[13]) != 0 else 0
    per_good_quality_6_months_no = np.round((len(firstScore[20]) / len(firstScore[14])) * 100, 2) if len(firstScore[14]) != 0 else 0

    print("Good Quality of Readmission within 28 Days (Yes) Percentage : ", per_good_quality_28_days_yes)
    print("Good Quality of Readmission within 28 Days (No) Percentage : ", per_good_quality_28_days_no)
    print("Good Quality of Readmission between 28 Days to 3 Months (Yes) Percentage : ", per_good_quality_3_months_yes)
    print("Good Quality of Readmission between 28 Days to 3 Months (No) Percentage : ", per_good_quality_3_months_no)
    print("Good Quality of Readmission between 3 Months to 6 Months (Yes) Percentage : ", per_good_quality_6_months_yes)
    print("Good Quality of Readmission between 3 Months to 6 Months (No) Percentage : ", per_good_quality_6_months_no)

    # Print average support and confidence for each category
    print("Average Support for Readmission within 28 Days (Yes) : ", firstScore[21])
    print("Average Confidence for Readmission within 28 Days (Yes) : ", firstScore[22])
    print("Average Support for Readmission within 28 Days (No) : ", firstScore[23])
    print("Average Confidence for Readmission within 28 Days (No) : ", firstScore[24])
    print("Average Support for Readmission between 28 Days to 3 Months (Yes) : ", firstScore[25])
    print("Average Confidence for Readmission between 28 Days to 3 Months (Yes) : ", firstScore[26])
    print("Average Support for Readmission between 28 Days to 3 Months (No) : ", firstScore[27])
    print("Average Confidence for Readmission between 28 Days to 3 Months (No) : ", firstScore[28])
    print("Average Support for Readmission between 3 Months to 6 Months (Yes) : ", firstScore[29])
    print("Average Confidence for Readmission between 3 Months to 6 Months (Yes) : ", firstScore[30])
    print("Average Support for Readmission between 3 Months to 6 Months (No) : ", firstScore[31])
    print("Average Confidence for Readmission between 3 Months to 6 Months (No) : ", firstScore[32])

    print("Total Execution Time : ", total_time)
else:
    print("Invalid Entry")
'''


# Auto-Encoder Rule Mining
print("......Auto-Encoder Rule Mining Starts here.............")
for i in range(nbRun):
    NN = ARMAE(dataSize, maxEpoch=nbEpoch, batchSize=batchSize, learningRate=learningRate, likeness=likeness, minSupp=minSupp_AE, minConf=minConf)
    dataLoader = NN.dataPretraitement(data)  # Preprocess the data for training
    t1 = time.time()  # Start timing the training
    if not isLoadedModel:
        NN.train(dataLoader, modelPath)  # Train the model
    else:
        NN.load(encoderPath, decoderPath)  # Load the pretrained model
    t2 = time.time()  # End timing the training
    timeTraining = t2 - t1  # Calculate training time
    print(f"Total Time Taken in Training is: {timeTraining}")
    path = os.path.join(ARMAEResultsPath, datasetName + str(i) + '.csv')
    timeCreatingRule, timeComputingMeasure = NN.generateRules(data, numberOfRules=numberOfRules, nbAntecedent=nbAntecedents_AE,
                                                              path=path, minSupp=minSupp_AE, minConf=minConf)
    t3 = time.time()  # End timing the rule generation
    timeRuleGen = t3 - t2  # Calculate rule generation time
    print(f"Total Time Taken in Generating Rules is: {timeRuleGen} using ARE-AE")

    timeTraining = t2 - t1
    times.append([timeTraining, timeCreatingRule, timeComputingMeasure])
    nnR = pd.read_csv(path)
    total_rules_generated = len(nnR)
    print(f"Total Number of Rules Generated by ARMAE: {total_rules_generated}")

    # Print the number of rules generated for each consequent
    try:
        consequent_val = nnR["consequent"].value_counts().keys()[0]
        consequent_count = nnR["consequent"].value_counts()[consequent_val]
        print(f"Number of Rules Generated for {consequent_val} are {consequent_count}")
    except:
        pass
    try:
        consequent_val = nnR["consequent"].value_counts().keys()[1]
        consequent_count = nnR["consequent"].value_counts()[consequent_val]
        print(f"Number of Rules Generated for {consequent_val} are {consequent_count}")
    except:
        pass

    # Calculate lift for the generated rules
    def calculate_lift(support, confidence):
        return confidence / support

    nnR['Lift'] = calculate_lift(nnR['support'], nnR['confidence'])

    # Filter good quality rules generated by autoencoder
    good_quality_rules_autoencoder = nnR[(nnR['Lift'] > 1.2) &
                                         (nnR['support'] > 0.01) &
                                         (nnR['confidence'] > 0.6)]

    # Separate rules based on different time frames for readmission
    consequents_1_autoencoder = "('label_1',)"
    readmission_30_days_yes_autoencoder = nnR[nnR['consequent'].astype(str) == consequents_1_autoencoder]
    good_quality_rules_readmission_30_days_yes = good_quality_rules_autoencoder[
        good_quality_rules_autoencoder['consequent'].astype(str) == consequents_1_autoencoder]

    consequents_2_autoencoder = "('label_0',)"
    readmission_30_days_no_autoencoder = nnR[nnR['consequent'].astype(str) == consequents_2_autoencoder]
    good_quality_rules_readmission_30_days_no = good_quality_rules_autoencoder[
        good_quality_rules_autoencoder['consequent'].astype(str) == consequents_2_autoencoder]

    

    total_length_autoencoder = len(nnR['Lift'])

    # Calculate percentages for good quality rules in autoencoder results
    good_quality_rules_30_days_yes_autoencoder_percentage = (len(good_quality_rules_readmission_30_days_yes) / total_length_autoencoder) * 100 if total_length_autoencoder != 0 else 0
    good_quality_rules_30_days_no_autoencoder_percentage = (len(good_quality_rules_readmission_30_days_no) / total_length_autoencoder) * 100 if total_length_autoencoder != 0 else 0
    
    # Save good quality rules generated by autoencoder to CSV
    path_autoencoder_quality = os.path.join(overallResultsPath, "readmission_30_days_ARMAE_good_quality_rules.csv")
    good_quality_rules_autoencoder.to_csv(path_autoencoder_quality, index=False)

    nnAvgSupp = np.mean(nnR['support'])
    nnAvgConf = np.mean(nnR['confidence'])
    print('support average ARM-AE : ', nnAvgSupp)
    print('confidence average ARM-AE : ', nnAvgConf)
    print("Total Number Of Readmission within 30 Days (Yes) Rules AutoEncoder : ", len(readmission_30_days_yes_autoencoder))
    print("Total Number Of Readmission within 30 Days (No) Rules AutoEncoder : ", len(readmission_30_days_no_autoencoder))
    print("Total Number Of Good Readmission within 30 Days (Yes) Rules AutoEncoder : ", len(good_quality_rules_readmission_30_days_yes))
    print("Total Number Of Good Readmission within 30 Days (No) Rules AutoEncoder : ", len(good_quality_rules_readmission_30_days_no))
    print("Good Quality of Readmission within 30 Days (Yes) Percentage : ", good_quality_rules_30_days_yes_autoencoder_percentage)
    print("Good Quality of Readmission within 30 Days (No) Percentage : ", good_quality_rules_30_days_no_autoencoder_percentage)

    # Print average support and confidence for each category in autoencoder results
    avg_support_30_days_yes_autoencoder = np.mean(readmission_30_days_yes_autoencoder['support']) if len(readmission_30_days_yes_autoencoder) > 0 else 0
    avg_confidence_30_days_yes_autoencoder = np.mean(readmission_30_days_yes_autoencoder['confidence']) if len(readmission_30_days_yes_autoencoder) > 0 else 0
    avg_support_30_days_no_autoencoder = np.mean(readmission_30_days_no_autoencoder['support']) if len(readmission_30_days_no_autoencoder) > 0 else 0
    avg_confidence_30_days_no_autoencoder = np.mean(readmission_30_days_no_autoencoder['confidence']) if len(readmission_30_days_no_autoencoder) > 0 else 0

    print("Average Support for Readmission within 30 Days (Yes) AutoEncoder : ", avg_support_30_days_yes_autoencoder)
    print("Average Confidence for Readmission within 30 Days (Yes) AutoEncoder : ", avg_confidence_30_days_yes_autoencoder)
    print("Average Support for Readmission within 30 Days (No) AutoEncoder : ", avg_support_30_days_no_autoencoder)
    print("Average Confidence for Readmission within 30 Days (No) AutoEncoder : ", avg_confidence_30_days_no_autoencoder)
