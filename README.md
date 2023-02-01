# nfl-linemen
For NFL Big Data Bowl 2023 Kaggle Competition

Summary: Participated in the 2023 NFL Big Data Bowl attempting to make a metric around NFL defensive lineman pursuit - basically what percentage of their
movement and force is going towards the ball over the course of the play.  Other than something sexy that reflected top rushers, it really did not give much insight.

Repo includes the following Jupyter notebooks where functions were developed and and tested:
1. Data Review [data_review.ipynb] - My cleaning pipeline applied to the NFL data provided.
2. Creating a Base Playframe [base_play_frame_workbook.ipynb] - For a given player on a given play, get their movement details and integrate with play-level stats and outcomes.
3. Create Unique Metrics [metrics_build_workbook.ipynb] - Creates metrics around force and movement vectors.  Also includes graphing functions.
4. Aggregating the Metrics for plays, teams, players, etc. [metrics_use_workbook.ipynb] - Allows use of passrush efficiency and force metrics for comparison.
5. EDA and Modeling ['EDA and Modeling.ipynb'] - Most of this is draft work.

From those Jupyter notebooks, the following .py files were created that reflect the functions developed with them:
1. nfl_acquire_and_prep.py
2. nfl_frame_builder.py
3. nfl_build_metrics.py
4. nfl_use_metrics.py

Overall, this is just a lot of junk, but if you want to learn about a cool, different metric based off of a pass rushers velocity (as well as force)
component towards the ball over the course of a play, reach out and I can walk you through this!
