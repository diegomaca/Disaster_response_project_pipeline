# Disaster_response_project_pipeline

This project aims to develop a complete pipeline, from Extract-Transform-Load (ETL) to deployment, for a classification model capable of predicting 36 binary problems. Each binary problem corresponds to a category associated with a message written on a social network. The pipeline leverages the Python libraries sklearn, logging, sqlite, and NLTK.

# Installation
To run this project, please ensure that you have the following libraries installed:

1. sklearn
2. logging
3. sqlite
4. NLTK

# Usage
To execute the ETL pipeline, which cleans the data and stores it in a database, run the following command:

* python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

To run the ML pipeline, which trains the classifier and saves it, use the following command:

* python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

# Deployment
After running the ETL and ML pipelines, you will have a trained classifier stored as classifier.pkl. This classifier can be deployed as an API capable of associating categories with new messages and displaying the results graphically in real-time.

# Contributing
Contributions to this project are welcome. If you have any suggestions or would like to add new features, please feel free to submit a pull request.

# License
This project is licensed under the MIT License. You are free to modify, distribute, and use the code as permitted by this license.

# Acknowledgments
1. The contributions of the open-source community and the developers of the libraries used in this project.
2. Udacity data scientis course. 

