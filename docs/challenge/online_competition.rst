Online Competition
==================

The online competition takes place every semester for the students of the "Autonomous Drone Racing" class at TUM. Students compete against each other in the level 3 scenario. 

.. note::
    The competition is useful to get an idea of how fast your controller is compared to other teams. It is **not** part of your grade or used to evaluate your final project submission.

Leaderboard
~~~~~~~~~~~
We use Kaggle to host the online competition. Students can submit their controllers to the competition by forking the repository and pushing their changes to the main branch. The best submission will be used for the team ranking in the online leaderboard.

.. note::
    Link to the current leaderboard.

.. note::
    The leaderboard is reset every semester. Participation is by invitation only. If you have not received the invite link, please contact the instructors.

Signing Up
----------
To participate in the online competition, follow these steps:

#. Fork the lsy_drone_racing repository to your own GitHub account.

#. Sign up for a Kaggle account if you don't already have one.

#. Join the competition using the invite link provided by the instructors.

#. Go to the competition page on Kaggle and accept the rules.


Submitting Your Controller
--------------------------
After you have joined the competition, you can start implementing your controller. We provide several controller examples in the control module. Once you have created your own controller, you should set the ``controller.file`` argument in the config files to your controller file. This will ensure that the controller is used by default when running the simulation, and also when submitting to the competition.

To submit your controller to the competition, you need to push your changes to the main branch of your forked repository. This will trigger a GitHub action to run your controller and submit the results to Kaggle. This automation requires you to set up your Kaggle credentials as GitHub repository secrets. You can do that by following these steps:

#. Go to your forked repository's Settings > Secrets and variables > Actions
#. Add two new repository secrets:
    * Name: KaggleUsername, Secret: Your Kaggle username
    * Name: KaggleKey, Secret: Your Kaggle API key

7. When you're ready to submit, push your changes to the main branch of your forked repository.

8. A GitHub action will automatically run, testing your implementation and submitting the results to Kaggle.

9. You can check the progress of your submission in the Actions tab of your repository.

10. The competition will use your fastest average lap time across all submissions.

Note: Kaggle limits submissions to 100 per day. The GitHub action caches dependencies after the first run, so subsequent runs will be faster.

If you need to add additional packages, update the environment.yaml file in your repository.

For more detailed information on your performance, check the test output in the GitHub action logs.
