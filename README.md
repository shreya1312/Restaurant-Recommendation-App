# Restaurant Recommendation System

## Problem Statement:
	
*We will develop a Restaurant Recommendation System and use a data driven solution to solve the problem of finding suitable locations to open restaurants*

## Environment Setup:

To run the code with different datasets, follow these instructions:

* Ensure that Python is installed on your computer.
* Set up a virtual environment to isolate the project dependencies (optional but recommended).
* Install the required libraries by running the following command:

```
pip install -r requirements.txt
```

## Dataset Preparation:

The code supports both CSV file input and manual input through the user interface (UI).
* If using a CSV file, ensure it is in the correct format and upload it.
* If opting for manual input, launch the application and follow the on-screen instructions to enter the restaurant data.
* If no user input is provided, the application uses zomato.csv file by default.

## Running the Application:

* Open a terminal or command prompt and navigate to the project directory.
* Execute the following command to start the Streamlit application:

```
cd src/phase3/; streamlit run restaurant-app.py
```

## Interacting with the UI:

* Access the application through the provided local URL, such as http://localhost:8501, in your web browser.
* Use the UI to upload a CSV file or manually enter restaurant data as required.
* Click on the "Predict" button in the sidebar to obtain restaurant location recommendations