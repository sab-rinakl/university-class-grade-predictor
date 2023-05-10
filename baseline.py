import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import mean_squared_error

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


def find_category(subject_code):
    for category, subjects in subject_dict.items():
        for subject, subject_codes in subjects.items():
            if subject_code in subject_codes:
                return category
    return None


def calculate_weighted_average(student_grades, predicted_class, c):
    predicted_class_info = predicted_class.split('-')
    predicted_class_subject = predicted_class_info[0]
    predicted_class_difficulty = int(predicted_class_info[1])
    weighted_sum = 0
    weight_sum = 0
    for class_name, grade in student_grades.items():
        class_info = class_name.split('-')
        class_subject = class_info[0]
        class_difficulty = int(class_info[1])

        numerical_grade = 0
        if pd.isna(grade) or grade == "WU" or grade == "I" or grade == "W":
            numerical_grade = gpa[c]
        else:
            _, numerical_grade = grade_dict[grade]

        if (class_difficulty % 100) > (predicted_class_difficulty % 100):
            class_diff_weight = 1.75
        elif class_difficulty == predicted_class_difficulty:
            class_diff_weight = 1.0
        else:
            class_diff_weight = 0.25
        if predicted_class_subject.lower() in class_subject.lower():
            class_sim_weight = 1.75
        elif find_category(predicted_class_subject.lower()) == find_category(class_subject.lower()):
            class_sim_weight = 1.0
        else:
            class_sim_weight = 0.25
        class_weight = class_diff_weight * class_sim_weight
        weighted_sum += numerical_grade * class_weight
        weight_sum += class_weight

    weighted_average = weighted_sum / weight_sum
    calc_predicted_grade = min(4.0, max(0.0, weighted_average))
    predicted_letter_grade = None
    for grade, (lower_bound, upper_bound) in grade_dict.items():
        if lower_bound < calc_predicted_grade <= upper_bound:
            predicted_letter_grade = grade
            break
    return predicted_letter_grade


parser = argparse.ArgumentParser(description='Predict the grade of a student in given classes')
parser.add_argument('filename', type=str, help='the filename of the csv file containing student grades')
parser.add_argument('predicted_classes', nargs='+', type=str, help='the names of the classes to predict the grade for')
args = parser.parse_args()

df = pd.read_csv(args.filename)

for predicted_class in args.predicted_classes:
    if predicted_class not in df.columns:
        raise ValueError(f"{predicted_class} not found in the CSV file. Please specify a class from the file.")

df.set_index('seat number', inplace=True)
gpa = df.iloc[:, -1].tolist()
df = df.iloc[:, :-1]
student_grades_dict = {}

for predicted_class in args.predicted_classes:
    df_class = df.dropna(subset=[predicted_class])
    real_grades = df_class[predicted_class]
    real_grades_list = []
    predicted_grades_list = []
    df_class = df_class.drop(columns=[predicted_class])
    counter = 0
    for index, row in df_class.iterrows():
        student_grades = row.to_dict()
        predicted_grade = calculate_weighted_average(student_grades, predicted_class, counter)
        real_grade = real_grades.loc[index]
        if pd.isna(real_grade) or real_grade == "WU" or real_grade == "I" or real_grade == "W":
            continue

        real_grades_list.append(grade_dict[real_grade][1])
        predicted_grades_list.append(grade_dict[predicted_grade][1])

        if index in student_grades_dict:
            student_grades_dict[index][predicted_class] = {'actual_grade': real_grade,
                                                           'predicted_grade': predicted_grade}
        else:
            student_grades_dict[index] = {
                predicted_class: {'actual_grade': real_grade, 'predicted_grade': predicted_grade}}

        counter = counter + 1

        print(f"Student {index} in {predicted_class}:")
        print(f"Predicted grade: {predicted_grade}")
        print(f"Real grade: {real_grade}")
        print()

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

