# a79ai_assignment



# HCP Dashboard and Chatbot

This project is a Streamlit application that shows a dashboard for healthcare providers (HCPs) and also provides a chatbot that can answer questions about HCPs, claims and prescriptions.

## Features

1. **Chatbot Assistant**

   * Ask questions about HCPs, patients, claims and prescriptions.
   * Gives answers from data using Q and A agent.
   * Provides related insights using Insight agent.

2. **Calendar**

   * A monthly calendar is shown in the chatbot page.
   * It highlights meetings and calls scheduled in the future.

3. **Notes Section**

   * A space to add and view HCP insights.
   * Useful for keeping track of observations, action points and reminders.
   * This Notes is created Randomly based on sample inputs. But with real data the same can be created after performing multiple analysis across various data sources.

4. **Dashboard**

   * Shows details of the selected HCP like name, ID, address and specialty.
   * Displays total patients, diagnosed patients and treated patients.
   * Charts for patient count by gender and by months.
   * Charts for claims by gender and by months.
   * Charts for patients by insurance and by brand.
   * Prescription data (NRX and TRX).
   * Call progress with milestones and completion alerts.


## How it works

* The data is loaded from CSV file.
* Chroma database is built with embeddings from Azure OpenAI.
* The chatbot uses the stored data to answer questions and give insights.
* The dashboard uses Plotly charts to show visual information.
* Streamlit is used to create an interactive web application.
* Calendar and notes make it easy to plan activities and store insights about HCPs.

## Requirements

* Python 3.9 or above
* Streamlit
* Pandas
* Plotly
* LangChain and related libraries
* Azure OpenAI credentials
* ChromaDB

## How to run

1. Clone the project or copy the files.
2. Install the required Python libraries using pip.
3. Set environment variables in a .env file (Azure OpenAI keys and Excel file path).
4. Run the app:

   ```bash
   streamlit run streamlit_app.py
   ```
5. Open the link shown in the terminal to use the app.

## Data

* `hcp_dashboard_data.csv` contains the data for the dashboard and chatbot.
---
