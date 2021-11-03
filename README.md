1.	Overview 

Crash triage app is developed to find the similar crashes/bugs already resolved in the system. App is trained on a bugs dataset using TFIDF so when a new bug is reported it is converted into a word vector and matched against existing bugs to find the similarity and then top 10 similar bugs are reported ranked based on cosine similarity, most similar bug will be ranked 1st.

2.	Dataset for training

Make sure your training csv contains bugs in below format.https://github.com/nandansamant/triage-heroku/blob/main/classifier_data_10_sample.csv
 
3.	Architecture

Dataset is fed to triage_train utility which uses TFIDF and converts each bug description into word vec. It then pickles all word vecs and input bugs into pickle files. These pickle files are fed to next triage_find utility which provides command line intf to take bug description as input, creates word vec of bug description and performs cosine similarity with input bugs and finds top ten most similar bugs and their respective developers who fixed them. This functionality is also hooked up to flask to provide user 
interface. App is also dockerized for portability.

4.	Live deployment

App is deployed on Heroku and live @ https://triage-capstone.herokuapp.com/ 

5.	References 

https://www.kaggle.com/crawford/deeptriage 

6. Documentation 

https://github.com/nandansamant/triage-heroku/blob/main/crash-triage.docx

7. Execution

      Step #1 - Train model using csv file which contains bug description and dev as columns (refer to classifier_data_10_sample.csv for the format)

      run using below command
      >> python triage_train.py -c classifier_data_10.csv

      once the model is trained in above step it'll generate pickle files which will be used in Step#2

      Step #2 - Find similar bugs and corresponding dev 

      run using below command
      >> python triage_find.py -b .\bug_file.txt
