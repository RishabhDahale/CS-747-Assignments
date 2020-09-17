instance=('../instances/i-1.txt' '../instances/i-2.txt' '../instances/i-3.txt')
algorithms=('thompson-sampling' 'thompson-sampling-with-hint')
horizon=(100 400 1600 6400 25600 102400)

i=1

for in in "${instance[@]}"; do
  for al in "${algorithms[@]}"; do
    for rs in {0..49}; do
      for hz in "${horizon[@]}"; do
        echo "Test $i"
        python bandit.py --instance "$in" --algorithm "$al" --randomSeed "$rs" --epsilon 0.02 --horizon "$hz"
        i=$((i + 1))
      done
    done
  done
done

#python bandit.py --instance ../instances/i-1.txt --algorithm thompson-sampling-with-hint --randomSeed 1 --epsilon 0.02 --horizon 30
