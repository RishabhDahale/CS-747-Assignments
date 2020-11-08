seeds=20
moves=4
algo=("sarsa" "expsarsa" "qlearning")
# Simple windy-gridworld
i=1
for seed in {1..seeds}; do
  for al in "${algo[@]}"; do
    echo "Simple Gridworld $i"
    python main.py --seed "$seed" --algo "$al" --$moves "$moves"
    i=$((i + 1))
  done
done

# Kings move
moves=8
i=1
for seed in {1..seeds}; do
  for al in "${algo[@]}"; do
    echo "Kings Move Gridworld (without 9th action) $i"
    python main.py --seed "$seed" --algo "$al" --$moves "$moves"
    i=$((i + 1))
  done
done

moves=9
i=1
for seed in {1..seeds}; do
  for al in "${algo[@]}"; do
    echo "Kings Move Gridworld (with 9th action) $i"
    python main.py --seed "$seed" --algo "$al" --$moves "$moves"
    i=$((i + 1))
  done
done

# Stochastic wind
windDev=1
moves=8
i=1
for seed in {1..seeds}; do
  for al in "${algo[@]}"; do
    echo "Kings Move Gridworld (with 9th action) $i"
    python main.py --seed "$seed" --algo "$al" --$moves "$moves" --windDev "$windDev"
    i=$((i + 1))
  done
done

python plot.py