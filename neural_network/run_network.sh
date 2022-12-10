# test for different hidden_dim with random init
for hidden in 5 10 25 50 100
do
    python3 network.py --epochs=15 --gamma=.1 --d=.2 --init='random' --hidden_dim=$hidden
done

# test for different hidden_dim with zero init
for hidden in 5 10 25 50 100
do
    python3 network.py --epochs=15 --gamma=.04 --d=.1 --init='zero' --hidden_dim=$hidden
done

