DATA_EMB_DIC1 = {
    'bonanza': (7919,1973),
    'house1to10': (515, 1281),
    'senate1to10': (145, 1056),
    'review': (182, 304)
}

DATA_EMB_DIC = {**DATA_EMB_DIC1}
for k in DATA_EMB_DIC1:
    for i in range(1, 6):
        DATA_EMB_DIC.update({
            f'{k}-{i}': DATA_EMB_DIC1[k]
        })

