import pandas as pd
import numpy as np
import argparse

from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder
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


def matrix_completion_collaborative_filtering(M, k):
    U, s, VT = np.linalg.svd(M, full_matrices=False)
    S = np.zeros((U.shape[1], VT.shape[0]))
    S[:k, :k] = np.diag(s[:k])
    M_hat = U.dot(S).dot(VT)
    return M_hat


# define a function to find the category for a given subject code
def find_category(subject_code):
    for category, subjects in subject_dict.items():
        for subject, subject_codes in subjects.items():
            if subject_code in subject_codes:
                return category
    return None


# New function to preprocess the data
def preprocess_data(df, predicted_class):
    # Drop rows with missing grades for the specified class
    df = df.dropna(subset=[predicted_class])

    # Store the real grades for the predicted class
    real_grades = df[predicted_class]

    # Drop the predicted class column for further processing
    df = df.drop(columns=[predicted_class])

    # One-hot encode the grades
    enc = OneHotEncoder()
    encoded_grades = enc.fit_transform(df)

    return encoded_grades, real_grades


# define the command line arguments
parser = argparse.ArgumentParser(description='Predict the grade of a student in a given class')
parser.add_argument('filename', type=str, help='the filename of the csv file containing student grades')
parser.add_argument('predicted_classes', type=str, nargs='+', help='the name of the class(es) to predict the grades for')

# parse the command line arguments
args = parser.parse_args()

# read the csv file into a pandas dataframe
df = pd.read_csv(args.filename)

# Check if all predicted_classes are in the columns of the DataFrame
missing_classes = set(args.predicted_classes) - set(df.columns)
if missing_classes:
    raise ValueError(f"{missing_classes} not found in the CSV file. Please specify a class from the file.")

# set the index of the dataframe to the 'seat number' column
df.set_index('seat number', inplace=True)

# select all rows and all columns except for the last one
df = df.iloc[:, :-1]

# Replace blank or empty values with NaN
df.replace("", np.nan, inplace=True)
df.replace(" ", np.nan, inplace=True)
df.replace("WU", np.nan, inplace=True)
df.replace("W", np.nan, inplace=True)
df.replace("I", np.nan, inplace=True)

# Define a list to store the predicted grades for each class
predicted_grades_list = []
student_grades_dict = {}
df = df.reset_index()

for predicted_class in args.predicted_classes:
    # Preprocess the data
    encoded_grades, real_grades = preprocess_data(df, predicted_class)

    # Convert real letter grades to numerical grades
    real_grades_numerical = np.array([grade_dict[grade][1] for grade in real_grades])

    # Perform matrix completion using user-based collaborative filtering
    encoded_grades_completed = matrix_completion_collaborative_filtering(encoded_grades.toarray(), k=7)

    # Compute the average grade for each class
    average_grade = np.nanmean(encoded_grades_completed, axis=0)

    # Compute the predicted grades, ensure they remained scaled to gpa
    predicted_grades_numerical = (encoded_grades_completed @ average_grade) / 2

    # Convert numerical grades back to letter grades
    predicted_grades = []
    for numerical_grade in predicted_grades_numerical:
        predicted_grade = "A"
        for grade, (lower_bound, upper_bound) in grade_dict.items():
            if lower_bound < numerical_grade <= upper_bound:
                predicted_grade = grade
                break

        predicted_grades.append(predicted_grade)

    # Append the predicted grades to the list
    predicted_grades_list.append(predicted_grades)

    # Store the actual and predicted grades for each student in the current class
    for index, (real_grade, predicted_grade) in enumerate(zip(real_grades, predicted_grades)):
        seat_num = df['seat number'][index]
        if seat_num not in student_grades_dict:
            student_grades_dict[seat_num] = {}
        student_grades_dict[seat_num][predicted_class] = {'actual_grade': real_grade,
                                                          'predicted_grade': predicted_grade}

    # Calculate the RMSE and print the actual and predicted grades for each student in the current class
    real_grades_list = list(real_grades_numerical)
    rmse = mean_squared_error(real_grades_list, predicted_grades_numerical, squared=False)

    seat_nums = list(df['seat number'])

    # print(f"Actual and predicted grades for {predicted_class}:")
    # for index, seat_num in enumerate(seat_nums):
    #     if seat_num in student_grades_dict and predicted_class in student_grades_dict[seat_num]:
    #         print(f"Student {seat_num}:")
    #         print(f"Predicted grade: {student_grades_dict[seat_num][predicted_class]['predicted_grade']}")
    #         print(f"Real grade: {student_grades_dict[seat_num][predicted_class]['actual_grade']}")
    #         print()

    print(f"Root Mean Squared Error for {predicted_class}: {rmse:.2f}")

total = 0
correct = 0
for student in student_grades_dict:
    # print(student)
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
    # print(max_actual_class, ":", max_actual_grade)
    # print(max_pred_class, ":", max_pred_grade)
    total = total + 1
    if max_actual_class == max_pred_class:
        correct = correct + 1

print("Predictions Correct:", (correct/total)*100, "%")
