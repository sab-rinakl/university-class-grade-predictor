# university-class-grade-predictor
Grade predictor project for CSCI467


Dataset: https://www.kaggle.com/datasets/ssshayan/grades-of-students


Command line arguments format: python3 script.py dataset.csv class1 class2 class3 ...


Command line arguments I used:

python3 baseline.py train.csv PH-121 CS-105 HS-205 MT-331 CS-412

python3 baseline.py dev.csv PH-121 CS-105 HS-205 MT-331 CS-412

python3 baseline.py test.csv PH-121 CS-105 HS-205 MT-331 CS-412

python3 linear_regression.py train.csv PH-121 CS-105 HS-205 MT-331 CS-412

python3 linear_regression.py dev.csv PH-121 CS-105 HS-205 MT-331 CS-412

python3 linear_regression.py test.csv PH-121 CS-105 HS-205 MT-331 CS-412

python3 matrix_completion.py train.csv PH-121 CS-105 HS-205 MT-331 CS-412

python3 matrix_completion.py dev.csv PH-121 CS-105 HS-205 MT-331 CS-412

python3 matrix_completion.py test.csv PH-121 CS-105 HS-205 MT-331 CS-412

