import sys
import csv
import toml
from pathlib import Path

actor = sys.argv[1]
csv_path = Path(__file__).parents[1] / "student_code/evaluation.csv"
toml_file = Path(__file__).parents[0] / "leaderboard.toml"

# 1. Read the time from the CSV
with open(csv_path, "r") as f:
    reader = csv.reader(f)
    row = next(reader)
    new_time = float(row[0])

# 2. Load the TOML
leaderboard = toml.load(toml_file)

# 3. Find the group by github username and update
for group_id, data in leaderboard.items():
    if data.get("github") == actor:
        current_time = data.get("time", float("nan"))

        if new_time < current_time:
            leaderboard[group_id]["time"] = new_time
            print(f"🎉 New personal best for {group_id}: {new_time}!")

        leaderboard[group_id]["submissions"] += 1
        break  # Stop searching once we found the team

# 4. Save the TOML
with open(toml_file, "w") as f:
    toml.dump(leaderboard, f)

print(f"✅ Updated leaderboard.toml for {actor}")
