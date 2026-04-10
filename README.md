# Dataset Bias Diagnostic Tool

## Overview
This repository contains an interactive visualization dashboard designed to assist in understanding dataset bias in Machine Learning predictions. Built with Streamlit, the tool allows users to actively introduce **Representation Bias** (starving demographics) and **Omitted Variable Bias** (dropping features) into a training dataset, and immediately visualize the downstream effects on a Logistic Regression model.

The tool compares a baseline model against a custom-biased model using standard ML evaluations alongside metrics specifically suited for imbalanced data, such as the Matthews Correlation Coefficient (MCC).

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



