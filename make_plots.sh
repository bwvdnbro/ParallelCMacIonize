#! /bin/bash

plot_tasks=
plot_costs=

while [ "$1" != "" ]
do
  case $1 in
    -t | --tasks ) plot_tasks=1;;
    -c | --costs ) plot_costs=1;;
  esac
  shift
done

# neutral fraction result
echo "Plotting physical result..."
python plot_neutral_fraction_profile.py
echo "Done."

# full task plot timeline
echo "Plotting full run task plot..."
python plot_full_timeline.py
echo "Done."

# per step task plot
if [ "$plot_tasks" = "1" ]
then
  echo "Plotting individual task plots per iteration..."
  for f in tasks_??.txt
  do
    python plot_tasks.py --name $f --labels
  done
  echo "Done."
fi

# per step cost distribution
if [ "$plot_costs" = "1" ]
then
  echo "Plotting cost distribution per iteration..."
  for f in costs_??.txt
  do
    python plot_cost_test.py $f
  done
  echo "Done."
fi
