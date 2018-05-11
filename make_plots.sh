#! /bin/bash

plot_tasks=
plot_costs=
plot_mpi=

while [ "$1" != "" ]
do
  case $1 in
    -t | --tasks ) plot_tasks=1;;
    -c | --costs ) plot_costs=1;;
    -m | --mpi ) plot_mpi=1;;
  esac
  shift
done

# neutral fraction result
echo "Plotting physical result..."
python scripts/plot_neutral_fraction_profile.py
echo "Done."

# full task plot timeline
echo "Plotting full run task plot..."
python scripts/plot_full_timeline.py --labels
echo "Done."

# memory stats
echo "Plotting memory statistics..."
python scripts/plot_memoryspace.py
python scripts/plot_queues.py
echo "Done."

# mpi stats
if [ "$plot_mpi" = "1" ]
then
  echo "Plotting MPI statistics..."
  python scripts/mpi_stats.py
  python scripts/plot_mpibuffer.py
  echo "Done."

  echo "Plotting communication statistics per iteration..."
  for f in messages_??.txt
  do
    python scripts/communication_stats.py --name $f --labels
  done
  echo "Done."
fi

# per step task plot
if [ "$plot_tasks" = "1" ]
then
  echo "Plotting individual task plots per iteration..."
  for f in tasks_??.txt
  do
    python scripts/plot_tasks.py --name $f --labels
  done
  echo "Done."
fi

# per step cost distribution
if [ "$plot_costs" = "1" ]
then
  echo "Plotting cost distribution per iteration..."
  for f in costs_??.txt
  do
    python scripts/plot_costs.py --name $f --labels
  done
  echo "Done."
fi
