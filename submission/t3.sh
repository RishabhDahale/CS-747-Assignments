instance=('../instances/i-1.txt' '../instances/i-2.txt' '../instances/i-3.txt')
#algorithms=('epsilon-greedy' 'ucb' 'thompson-sampling' 'kl-ucb')
algorithms=('epsilon-greedy')
horizon=(100 400 1600 6400 25600 102400)
randomSeeds=()
epsilon=(0.005 0.01 0.015 0.02 0.025 0.03 0.035 0.04 0.045 0.05 0.1 0.15)

i=1

for al in "${algorithms[@]}"; do
  for in in "${instance[@]}"; do
    for rs in {0..49}; do
      for hz in "${horizon[@]}"; do
      	for ep in "${epsilon[@]}"; do
 	       	echo "Test $i"
    	    python bandit.py --instance "$in" --algorithm "$al" --randomSeed "$rs" --epsilon "$ep" --horizon "$hz"
        	i=$((i + 1))
        done
      done
    done
  done
done

