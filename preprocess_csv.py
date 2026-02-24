import argparse
"""
This program takes a SAF csv and cleans it so it can later be correctly
read by pandas and then processed.
"""

parser = argparse.ArgumentParser(description="Program to fix given device data")

parser.add_argument("--path", type=str, required=True, help="Path to data csv")

args = parser.parse_args()

read_path = args.path

write_path = read_path[:-4]+"-Fix.csv"

write_file = open(write_path, "w")

with open(read_path, "r") as f:
    header = next(f)
    write_file.write(header.replace("\"", ""))
    for line in f:
        line_rep = line.replace("\"", "")
        day, date, data = line_rep.split(",", 2)
        date = "\"" + day + date + "\""
        data = data.replace("\"", "")
        fixed_line = date + "," + data
        write_file.write(fixed_line)