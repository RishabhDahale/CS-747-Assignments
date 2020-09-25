instance=('../instances/i-1.txt' '../instances/i-2.txt' '../instances/i-3.txt')
# instance=('../instances/i-2.txt' '../instances/i-3.txt')
algorithms=('thompson-sampling-with-hint' 'thompson-sampling')
# algorithms=('thompson-sampling')
horizon=(100 400 1600 6400 25600 102400)

i=1


for al in "${algorithms[@]}"; do
  for in in "${instance[@]}"; do
    for rs in {600..649}; do
      for hz in "${horizon[@]}"; do
        echo "Test $i"
        python bandit.py --instance "$in" --algorithm "$al" --randomSeed "$rs" --epsilon 0.02 --horizon "$hz"
        i=$((i + 1))
      done
    done
  done
done

#python bandit.py --instance ../instances/i-1.txt --algorithm thompson-sampling-with-hint --randomSeed 1 --epsilon 0.02 --horizon 30
