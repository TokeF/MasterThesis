
rho = range(2,200, 10)
thick = range(2, 20, 1)
thick = range(2, 100, 1)
layers = 3

f = open("denmark.mod", "w+")
for r1 in rho:
    for r2 in rho:
        for r3 in rho:
            for t1 in thick:
                for t2 in thick:
                    model = [r1, r2, r3, t1, t2]
                    for m in model:
                        f.write(str(m) + "\n")



