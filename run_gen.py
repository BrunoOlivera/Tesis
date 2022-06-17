psss = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
vsss = [5e-2, 1e-1, 2e-1, 3e-1, 5e-1, 8e-1]
gammas = [0.9, 0.99, 0.999, 1]
sigmas = [0.1, 1, 17]

for pss in psss:
    for vss in vsss:
        for gamma in gammas:
            for sigma in sigmas:
                print(f'start /B python run.py {pss} {vss} {gamma} {sigma} 5000 > output/output_{pss}_{vss}_{gamma}_{sigma}_5000')
