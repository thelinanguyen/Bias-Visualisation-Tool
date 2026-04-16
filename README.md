# Dataset Bias Diagnostic Tool

## Overview
This repository contains an interactive visualization dashboard designed to assist in understanding dataset bias in Machine Learning predictions. Built with Streamlit, the tool allows users to actively introduce **Representation Bias** (starving demographics) and **Omitted Variable Bias** (dropping features) into a training dataset, and immediately visualize the downstream effects on a Logistic Regression model.

The tool compares a baseline model against a custom-biased model using standard ML evaluations alongside metrics specifically suited for imbalanced data, such as the Matthews Correlation Coefficient (MCC).

<img width="2559" height="1296" alt="image" src="https://github.com/user-attachments/assets/4cf7fdb0-5170-4d90-a26e-96c90875f58e" />
<img width="2549" height="1336" alt="image" src="https://github.com/user-attachments/assets/8c0ddd11-5fa0-4c3e-9680-dfb956111ae2" />
<img width="2545" height="1312" alt="image" src="https://github.com/user-attachments/assets/56a695cd-c0f5-4ec9-a933-d462f2e428f5" />
<img width="2545" height="1318" alt="image" src="https://github.com/user-attachments/assets/1937ddd9-eae2-4f1a-ae0a-cce6eb632b6d" />


## Project Architecture
The project follows a modular design, separating the frontend UI from the backend logic:

* `src/app.py`: The main Streamlit dashboard and user interface.
* `src/data_preparation.py`: ETL pipeline fetching and cleaning the Adult Census dataset.
* `src/model.py`: Backend machine learning logic, pipeline creation, and metric calculations.
* `src/visualisation.py`: Helper functions for generating Plotly charts and confusion matrices.
* `requirements.txt`: Required Python dependencies.

## Installation

To run this project locally, you will need Python 3.8 or higher. 

1. Clone this repository to your local machine.
2. (Optional but recommended) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
3. Install the required packages:
    ```
    pip install -r requirements.txt

## Usage
To launch the interactive dashboard, run the following command from the root of the repository:

    streamlit run src/app.py

The application will automatically open in your default web browser.

## Data Source
This project utilizes the Adult Census Income dataset sourced from the UCI Machine Learning Repository.
https://archive.ics.uci.edu/dataset/2/adult




