# Disaster Response Project Pipeline
This project aims to develop an excellent pipeline, from Extract-Transform-Load (ETL) to deployment, for a classification model capable of predicting 36 binary problems. Each binary problem corresponds to a category associated with a message written on a social network. The pipeline leverages several powerful Python libraries, including sklearn, logging, sqlite, and NLTK.

# Installation
To run this project, please ensure that you have the following libraries installed:

1. sklearn
2. logging
3. sqlite
4. NLTK

# Usage
To execute the ETL pipeline, which cleans the data and stores it in a database, use the following command:

* python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

To run the ML pipeline, which trains the classifier and saves it, use the following command:

* python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

# Deployment
After running the ETL and ML pipelines, you will have a trained classifier stored as classifier.pkl. This classifier can be deployed as an API capable of associating categories with new messages and displaying the results graphically in real-time.

# Contributing
Contributions to this project are highly appreciated. If you have any suggestions or would like to add new features, please feel free to submit a pull request.

# License
This project is licensed under the MIT License. You are free to modify.

# Results

Check this link to view the process: https://github.com/diegomaca/Disaster_response_project_pipeline.git

# Acknowledgments

1. The open-source community and the developers of the libraries used in this project for their valuable contributions.
2. Udacity's data science course for providing guidance and inspiration.

# Author 

Diego Maca

Thank you for your interest in the Disaster Response Project Pipeline.
