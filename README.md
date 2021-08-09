# Learning-Augmented Algorithms for Online Subset Sum

This is the code for the paper "Learning-Augmented Algorithms for Online Subset Sum"

# Requirements

Python >= 3.6.11

Pytorch >= 1.7.0


# Adversary Train
To train the adversary network for each algorithm and output the bad instance set, run
```
bash train_adversary.sh
```
After training, the instance file for each algorithm will be created.

# Algorithm Test
To test the performance of each algorithm in its corresponding instance file, run

```
bash test.sh
```

