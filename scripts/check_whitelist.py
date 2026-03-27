import sys
import toml
from pathlib import Path

actor = sys.argv[1]
toml_file = Path("scripts/leaderboard.toml")
leaderboard = toml.load(toml_file)

# Search for the github username in the TOML
authorized = False
for group_id, data in leaderboard.items():
    if data.get("github") == actor:
        authorized = True
        break

if authorized:
    print(f"✅ User '{actor}' authorized. Proceeding...")
    sys.exit(0)
else:
    print(f"🛑 Unauthorized: User '{actor}' is not registered in leaderboard.toml.")
    sys.exit(1)
