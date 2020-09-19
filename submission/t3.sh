instance=('../instances/i-1.txt' '../instances/i-2.txt' '../instances/i-3.txt')
# instance=('../instances/i-1.txt')
#algorithms=('epsilon-greedy' 'ucb' 'thompson-sampling' 'kl-ucb')
algorithms=('epsilon-greedy')
horizon=(100 400 1600 6400 25600 102400)
randomSeeds=()
# epsilon=(0.002 0.004 0.006 0.008 0.01)
epsilon=(0.0001 0.005 0.25)

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

