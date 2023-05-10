import csv
import random

# Open the input and output files
with open('grades.csv', 'r') as input_file, \
        open('train.csv', 'w', newline='') as train_file, \
        open('dev.csv', 'w', newline='') as dev_file, \
        open('test.csv', 'w', newline='') as test_file:
    # Create CSV writer objects for the output files
    train_writer = csv.writer(train_file)
    dev_writer = csv.writer(dev_file)
    test_writer = csv.writer(test_file)

    # Create a CSV reader object for the input file
    reader = csv.reader(input_file)

    # Get the header row and write it to all three output files
    header_row = next(reader)
    train_writer.writerow(header_row)
    dev_writer.writerow(header_row)
    test_writer.writerow(header_row)

    # Read the remaining rows and shuffle them
    data = list(reader)
    random.shuffle(data)

    # Calculate the sizes of the train, dev, and test sets
    num_rows = len(data)
    num_train = int(num_rows * 0.8)
    num_dev = int(num_rows * 0.1)
    num_test = num_rows - num_train - num_dev

    # Write the rows to the output files
    train_writer.writerows(data[:num_train])
    dev_writer.writerows(data[num_train:num_train + num_dev])
    test_writer.writerows(data[num_train + num_dev:])
