# How to use
Each term, update the `leaderboard.toml` with the teams that have registered to whitelist them to the competition. After the term, the leaderboards have to be archived manually.

# Workflow
After a team is whitelisted, the workflow of the competition is as follows
1. Team pushes to the main branch in their repo
1. A `repository_dispatch` is sent to the central repo
1. The central repo clones the code of the team, runs the evaluation, and updates the leaderboard