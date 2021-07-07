# Disaster Response Pipeline Project

### About

It's important to know any disaster using people's message data. The ML model will be created and trained on the collected message data and used to predict a disaster category based on a message using NLP techniques

### HowTo Instructions:

1. Load and clean data csv files and store the cleaned version into sqlite database file on disk
```bash
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```

2. Train and optimize(hyperparameter search) the ML model on the cleaned data and evaluate its performance using precision, recall and f1 scores
```bash
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
```

3. Run simple Flask web app to visualize the data and to show how the trained ML model works
```bash
cd app/
python run.py
```

4. The Flask web app will be run on `http://localhost:3001/` for exploration


### File structure

- app/: 
  - templates/: The html page templates
  - run.py: The Flask webapp script
- data/: 
  - *.csv: The disaster category and message data in csv files
  - process_data.py: The data ETL pipeline script
- models/:
  - train_classifier.py: The ML model training script