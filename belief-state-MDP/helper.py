def mul(myList):
    # Multiply elements one by one
    result = 1
    for x in myList:
        result = result * x
    return result


def do(lst):
    mul1 = [mul(lst1) for lst1 in lst]
    sum1 = sum(mul1)
    return [item / sum1 for item in mul1]

#sanity check
print(do([[0.84, 0.2, 0.3, 1], [0.6, 0.8, 0.3, 1]]))
