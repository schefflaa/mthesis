# 📁 File Structure

<pre>
.
├── aws                         # AWS related stuff
│   ├── lambdas                 # AWS Lambda functions for data collection
│   ├── logfiles                # various Logfiles
│   └── utility                 # Utility functions for AWS
│
│── baselines                   # Baseline Models
│   └── ...
│
├── data                        # Main local Data Storage, e.g. Solar Power Data, Webcam Data, ...#
│   ├── poc_multiple            # Data for main experiments
│   └── ...
│
├── eda                         # Exploratory Data Analysis
│
├── eval                        # Evaluation of Models
│
├── models                      # Model Implementations
│
├── poc_mulithorizon            # Main Training Logic for Multihorizon Forecasting
│   ├── tblogs                  # Tensorboard Logs
│   └── ...
│
├── poc_multiple                # Main Training Logic
│   ├── tblogs                  # Tensorboard Logs
│   ├── utility                 # Tests, Utility Functions, ...
│   └── ...
│
├── utility                     # Main Utility Functions
│   ├── metrics                 # Evaluation Metrics for Model Performance
│   └── scripts                 # Scripts for Data Collection, Preprocessing, ...
│
├── .gitignore
├── <b>README.md</b>
└── requirements.txt
</pre>