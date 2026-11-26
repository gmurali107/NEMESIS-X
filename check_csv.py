
import csv
import os


file_path = "anomaly_alerts.csv"

if not os.path.exists(file_path):
	print("ERROR: anomaly_alerts.csv not found!")

else:
	with open(file_path, "r", newline="", encoding="utf-8-sig") as file:
		reader = csv.DictReader(file)

		print("\nCSV FILE SUCCESSFULLY OPENED")
		print("\nColumn names:")
		print(reader.fieldnames)

		print("\nFirst 5 records:")

		for index, row in enumerate(reader):
			if index >= 5:
				break

			print(row)
