Step #1 - Train model using csv file which contains bug description and dev as columns (refer to classifier_data_10_sample.csv for the format)

run using below command
python triage_train.py -c classifier_data_10.csv

Step #2 - Find similar bugs and corresponding dev 

run using below command
python triage_find.py -b .\bug_file.txt