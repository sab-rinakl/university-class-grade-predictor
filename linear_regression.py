import pandas as pd
import numpy as np
import argparse
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDRegressor
from sklearn.impute import KNNImputer

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


def preprocess_data(df, predicted_class):
    df = df.dropna(subset=[predicted_class])
    real_grades = df[predicted_class]
    df = df.drop(columns=[predicted_class])
    enc = OneHotEncoder()
    encoded_grades = enc.fit_transform(df)
    imputer = KNNImputer(n_neighbors=5, weights='uniform')
    encoded_grades_imputed = imputer.fit_transform(encoded_grades.toarray())
    return encoded_grades_imputed, real_grades


def train_linear_regression(encoded_grades, real_grades_numerical):
    model = SGDRegressor(max_iter=1000, learning_rate='constant', eta0=0.01)
    model.fit(encoded_grades, real_grades_numerical)
    return model


def find_category(subject_code):
    for category, subjects in subject_dict.items():
        for subject, subject_codes in subjects.items():
            if subject_code in subject_codes:
                return category
    return None


parser = argparse.ArgumentParser(description='Predict the grade of a student in a given class')
parser.add_argument('filename', type=str, help='the filename of the csv file containing student grades')
parser.add_argument('predicted_classes', type=str, nargs='+', help='the name(s) of the class(es) to predict the grades for')

args = parser.parse_args()

args = parser.parse_args()

df = pd.read_csv(args.filename)

missing_classes = set(args.predicted_classes) - set(df.columns)
if missing_classes:
    raise ValueError(f"{missing_classes} not found in the CSV file. Please specify classes from the file.")

df.set_index('seat number', inplace=True)
df = df.iloc[:, :-1]

df.replace("", np.nan, inplace=True)
df.replace(" ", np.nan, inplace=True)
df.replace("WU", np.nan, inplace=True)
df.replace("W", np.nan, inplace=True)
df.replace("I", np.nan, inplace=True)

predicted_grades_list = []
student_grades_dict = {}
df = df.reset_index()

for predicted_class in args.predicted_classes:
    encoded_grades, real_grades = preprocess_data(df, predicted_class)
    real_grades_numerical = np.array([grade_dict[grade][1] for grade in real_grades])
    model = train_linear_regression(encoded_grades, real_grades_numerical)
    predicted_grades_numerical = model.predict(encoded_grades)
    predicted_grades = []
    for numerical_grade in predicted_grades_numerical:
        predicted_grade = "A"
        for grade, (lower_bound, upper_bound) in grade_dict.items():
            if lower_bound < numerical_grade <= upper_bound:
                predicted_grade = grade
                break
        predicted_grades.append(predicted_grade)
    predicted_grades_list.append(predicted_grades)
    for index, (real_grade, predicted_grade) in enumerate(zip(real_grades, predicted_grades)):
        seat_num = df['seat number'][index]
        if seat_num not in student_grades_dict:
            student_grades_dict[seat_num] = {}
        student_grades_dict[seat_num][predicted_class] = {'actual_grade': real_grade,
                                                          'predicted_grade': predicted_grade}

    real_grades_list = list(real_grades_numerical)
    rmse = mean_squared_error(real_grades_list, predicted_grades_numerical, squared=False)
    seat_nums = list(df['seat number'])

    print(f"Actual and predicted grades for {predicted_class}:")
    for index, seat_num in enumerate(seat_nums):
        if seat_num in student_grades_dict and predicted_class in student_grades_dict[seat_num]:
            print(f"Student {seat_num}:")
            print(f"Predicted grade: {student_grades_dict[seat_num][predicted_class]['predicted_grade']}")
            print(f"Real grade: {student_grades_dict[seat_num][predicted_class]['actual_grade']}")

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
