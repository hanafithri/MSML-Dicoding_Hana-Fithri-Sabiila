import requests

data = {
    "Age": 56.0,
    "Gender": 22.0,
    "Marital_Status": 2.0,
    "Department": 1.0,
    "Job_Role": 3.0,
    "Job_Level": 1.0,
    "Monthly_Income": 2000.0,
    "Hourly_Rate": 40.0,
    "Years_at_Company": 3.0,
    "Years_in_Current_Role": 2.0,
    "Years_Since_Last_Promotion": 4.0,
    "Work_Life_Balance": 3.0,
    "Job_Satisfaction": 3.5,
    "Performance_Rating": 2.0,
    "Training_Hours_Last_Year": 20.0,
    "Overtime": 1.0,
    "Project_Count": 4.0,
    "Average_Hours_Worked_Per_Week": 45.0,
    "Absenteeism": 2.0,
    "Work_Environment_Satisfaction": 4.0,
    "Relationship_with_Manager": 4.0,
    "Job_Involvement": 3.0,
    "Distance_From_Home": 26.0,
    "Number_of_Companies_Worked": 2.0
}

response = requests.post(
    "http://127.0.0.1:5000/predict",
    json=data
)

print(response.json())