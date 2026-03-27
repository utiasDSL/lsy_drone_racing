import csv
import sys
import toml
from pathlib import Path

import numpy as np

actor = sys.argv[1]
csv_path = Path(__file__).parents[1] / "student_code/evaluation.csv"
toml_file = Path(__file__).parents[0] / "leaderboard.toml"

# Load data
with open(csv_path, "r") as f:
    reader = csv.reader(f)
    row = next(reader)
    new_time = float(row[0])
    new_success_rate = float(row[1])

leaderboard = toml.load(toml_file)

# Update leaderboard
new_best = False
for group_id, data in leaderboard.items():
    if data.get("github") == actor:
        current_time = data.get("time", np.inf)

        if new_time < current_time:
            leaderboard[group_id]["time"] = new_time
            leaderboard[group_id]["success_rate"] = new_success_rate
            leaderboard[group_id]["submissions"] += 1
            print(f"🎉 New personal best for {group_id}: {new_time}s!")
            new_best = True

        break  # Stop searching once we found the team

if not new_best:
    print(f"🛑 No new best ({new_time}s), won't submit to leaderboard.")
    sys.exit(0)  # Exits cleanly. The TOML file remains unchanged.

with open(toml_file, "w") as f:
    toml.dump(leaderboard, f)

print(f"✅ Updated leaderboard.toml for {actor}")
