import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import mean_squared_error

# define the grade conversion dictionary
grade_dict = {
    'A+': (4.0, 4.0),
    'A': (3.7, 4.0),
    'A-': (3.3, 3.7),
    'B+': (3.0, 3.3),
    'B': (2.7, 3.0),
    'B-': (2.3, 2.7),
    'C+': (2.0, 2.3),
    'C': (1.7, 2.0),
    'C-': (1.3, 1.7),
    'D+': (1.0, 1.3),
    'D': (0.0, 1.0),
    'F': (0.0, 0.0),
}


subject_dict = {
    "Math and Science": {
        "Mathematics": ["MT"],
        "Physics": ["PH"],
        "Chemistry": ["CY"],
        "Computer Science": ["CS"],
    },
    "Engineering": {
        "Electrical Engineering": ["EE"],
        "Mechanical Engineering": ["ME"],
        "Engineering Fundamentals": ["EF"],
    },
    "Humanities and Social Sciences": {
        "Humanities and Social Sciences": ["HS"]
    },
    "Communication": {
        "English Language": ["EL"],
        "Technical Communication": ["TC"],
    }
}


# define a function to find the category for a given subject code
def find_category(subject_code):
    for category, subjects in subject_dict.items():
        for subject, subject_codes in subjects.items():
            if subject_code in subject_codes:
                return category
    return None


# define the function to calculate the weighted average
def calculate_weighted_average(student_grades, predicted_class, c):
    predicted_class_info = predicted_class.split('-')  # split the predicted class into subject and number
    predicted_class_subject = predicted_class_info[0]  # get the subject of the predicted class
    predicted_class_difficulty = int(predicted_class_info[1])  # get the difficulty of the predicted class
    weighted_sum = 0
    weight_sum = 0
    for class_name, grade in student_grades.items():
        class_info = class_name.split('-')  # split the class name into subject and number
        class_subject = class_info[0]  # get the subject of the class
        class_difficulty = int(class_info[1])  # get the difficulty of the class

        numerical_grade = 0
        if pd.isna(grade) or grade == "WU" or grade == "I" or grade == "W":
            numerical_grade = gpa[c]
        else:
            # calculate the numerical grade
            _, numerical_grade = grade_dict[grade]

        # get the class difficulty weight
        if (class_difficulty % 100) > (predicted_class_difficulty % 100):
            class_diff_weight = 1.75
        elif class_difficulty == predicted_class_difficulty:
            class_diff_weight = 1.0
        else:
            class_diff_weight = 0.25
        # get the class similarity weight
        if predicted_class_subject.lower() in class_subject.lower():
            class_sim_weight = 1.75
        elif find_category(predicted_class_subject.lower()) == find_category(class_subject.lower()):
            class_sim_weight = 1.0
        else:
            class_sim_weight = 0.25
        # calculate the weight of the class
        class_weight = class_diff_weight * class_sim_weight
        # add to the weighted sum
        weighted_sum += numerical_grade * class_weight
        # add to the weight sum
        weight_sum += class_weight

    # calculate the weighted average
    weighted_average = weighted_sum / weight_sum
    # calculate the predicted grade
    calc_predicted_grade = min(4.0, max(0.0, weighted_average))
    # translate the numerical grade to letter grade
    predicted_letter_grade = None
    for grade, (lower_bound, upper_bound) in grade_dict.items():
        if lower_bound < calc_predicted_grade <= upper_bound:
            predicted_letter_grade = grade
            break
    return predicted_letter_grade


# define the command line arguments
parser = argparse.ArgumentParser(description='Predict the grade of a student in given classes')
parser.add_argument('filename', type=str, help='the filename of the csv file containing student grades')
parser.add_argument('predicted_classes', nargs='+', type=str, help='the names of the classes to predict the grade for')

# parse the command line arguments
args = parser.parse_args()

# read the csv file into a pandas dataframe
df = pd.read_csv(args.filename)

# Check if the predicted_classes are in the columns of the DataFrame
for predicted_class in args.predicted_classes:
    if predicted_class not in df.columns:
        raise ValueError(f"{predicted_class} not found in the CSV file. Please specify a class from the file.")

# set the index of the dataframe to the 'seat number' column
df.set_index('seat number', inplace=True)

gpa = df.iloc[:, -1].tolist()

# select all rows and all columns except for the last one
df = df.iloc[:,:-1]

# define an empty dictionary to store the actual and predicted grades for each student
student_grades_dict = {}

for predicted_class in args.predicted_classes:
    # Drop rows with missing grades for the specified class
    df_class = df.dropna(subset=[predicted_class])

    # Store the real grades for the predicted class
    real_grades = df_class[predicted_class]

    # Initialize empty lists for real and predicted grades
    real_grades_list = []
    predicted_grades_list = []

    # Drop the predicted class column for further processing
    df_class = df_class.drop(columns=[predicted_class])

    counter = 0
    # iterate over the rows of the dataframe
    for index, row in df_class.iterrows():
        # get the student's grades as a dictionary
        student_grades = row.to_dict()
        # calculate the weighted average for the student
        predicted_grade = calculate_weighted_average(student_grades, predicted_class, counter)
        # Get the real grade for the student in the predicted class
        real_grade = real_grades.loc[index]

        if pd.isna(real_grade) or real_grade == "WU" or real_grade == "I" or real_grade == "W":
            continue

        # Append the real and predicted grades to the lists
        real_grades_list.append(grade_dict[real_grade][1])
        predicted_grades_list.append(grade_dict[predicted_grade][1])

        # add the actual and predicted grades to the student_grades_dict
        if index in student_grades_dict:
            # if the student is already in the dictionary, add the predicted class and grades
            student_grades_dict[index][predicted_class] = {'actual_grade': real_grade,
                                                           'predicted_grade': predicted_grade}
        else:
            # if the student is not in the dictionary, create a new entry with the predicted class and grades
            student_grades_dict[index] = {
                predicted_class: {'actual_grade': real_grade, 'predicted_grade': predicted_grade}}

        counter = counter + 1

    # Calculate the RMSE between real and predicted grades
    rmse = mean_squared_error(real_grades_list, predicted_grades_list, squared=False)
    print(f"Root Mean Squared Error for {predicted_class}: {rmse:.2f}")

total = 0
correct = 0
for student in student_grades_dict:
    print(student)
    max_actual_grade = "F"
    max_pred_grade = "F"
    max_actual_class = ""
    max_pred_class = ""
    for predicted_class in args.predicted_classes:
        c = student_grades_dict[student].get(predicted_class)
        if c is not None:
            actual_grade = c.get("actual_grade")
            if actual_grade[0] <= max_actual_grade[0]:
                max_actual_grade = actual_grade
                max_actual_class = predicted_class
            pred_grade = c.get("predicted_grade")
            if pred_grade[0] <= max_pred_grade[0]:
                max_pred_grade = pred_grade
                max_pred_class = predicted_class
    print(max_actual_class, ":", max_actual_grade)
    print(max_pred_class, ":", max_pred_grade)
    total = total + 1
    if max_actual_class == max_pred_class:
        correct = correct + 1

print("Predictions Correct:", (correct/total)*100, "%")

