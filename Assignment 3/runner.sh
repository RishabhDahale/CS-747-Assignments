moves=4
algo=('sarsa' 'expsarsa' 'qlearning')
# Simple windy-gridworld
i=1
for al in "${algo[@]}"; do
  for seed in {1..20}; do
    echo "Simple Gridworld $i"
    python3 main.py --seed "$seed" --algo "$al" --moves "$moves"
    i=$((i + 1))
  done
done

# Kings move
algo=('sarsa')
moves=8
i=1
for seed in {1..20}; do
  for al in "${algo[@]}"; do
    echo "Kings Move Gridworld (8 actions) $i"
    python3 main.py --seed "$seed" --algo "$al" --moves "$moves"
    i=$((i + 1))
  done
done

moves=9
i=1
for seed in {1..20}; do
  for al in "${algo[@]}"; do
    echo "Kings Move Gridworld (9 actions) $i"
    python3 main.py --seed "$seed" --algo "$al" --moves "$moves"
    i=$((i + 1))
  done
done

# Stochastic wind
windDev=1
moves=8
i=1
for seed in {1..20}; do
  for al in "${algo[@]}"; do
    echo "Kings Move Gridworld stochastic (8 actions) $i"
    python3 main.py --seed "$seed" --algo "$al" --moves "$moves" --windDev "$windDev"
    i=$((i + 1))
  done
done

echo "Making Plots"
python3 plot.py

