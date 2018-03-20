#! /bin/bash

# neutral fraction result
python plot_intensity_profile.py

# full task plot timeline
python plot_full_timeline.py

# per step task plot
for f in tasks_??.txt
do
  python plot_tasks.py $f
done

# per step cost distribution
for f in costs_??.txt
do
  python plot_cost_test.py $f
done
