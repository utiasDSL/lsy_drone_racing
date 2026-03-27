from pathlib import Path
import toml

import numpy as np

toml_file = Path(__file__).parents[0] / "leaderboard.toml"
leaderboard = toml.load(toml_file)

### Sort teams
teams = []
for group_id, data in leaderboard.items():
    teams.append(
        {
            "name": data.get("name", "Unknown"),
            "time": data.get("time", np.inf),
            "submissions": data.get("submissions", 0),
        }
    )

teams.sort(key=lambda d: d["time"])

### Generate table
lines = ["| Rank | Team | Time | Submissions |", "| :---: | :--- | :--- | :---: |"]

for rank, team in enumerate(teams):
    time_val = team["time"]
    name = team["name"]
    subs = team["submissions"]

    # Handle formatting for unranked (nan) vs ranked teams
    if np.isinf(time_val):
        rank_str = "-"
        time_str = "N/A"
    else:
        medals = ["🥇 1", "🥈 2", "🥉 3"]
        rank_str = medals[rank] if rank < 3 else str(rank + 1)
        time_str = f"{time_val:.3f}"

    lines.append(f"| {rank_str} | {name} | {time_str} | {subs} |")

markdown_table = "\n".join(lines)

### Inject and store table
template = Path(__file__).parents[0] / "README_template.md"
with open(template, "r") as file:
    template_content = file.read()

final_readme = template_content.replace("leaderboard_placeholder", markdown_table)

readme = Path(__file__).parents[1] / "README.md"
with open("README.md", "w") as file:
    file.write(final_readme)

print("✅ README.md successfully regenerated!")
