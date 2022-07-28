psss = [1e-06, 5e-06, 1e-07, 5e-07, 1e-08, 5e-08, 1e-09, 3e-09, 5e-09, 7e-09]
vsss = [0.2, 0.1, 0.07, 0.05, 0.03, 0.01, 0.005]
epss = [0, 5, 10, 25]
sigmas = [5, 34, 136]
rbfs = [(41, 101), (82, 101), (164, 101), (328, 101), (656, 101), (250, 11), (500, 11)]

for pss in psss:
    for vss in vsss:
        for sigma in sigmas:
            for eps in epss:
                for rbf in rbfs:
                    print(f'sbatch -x node13 prototipo.sh {pss} {vss} {sigma} {eps} {rbf[1]} {rbf[0]}')
