def mul(myList):
    # Multiply elements one by one
    result = 1
    for x in myList:
        result = result * x
    return result

def do(a,b):
    a = mul(a)
    b = mul(b)
    sum1 =a + b
    return [a/sum1,b/sum1]


print(do([0.7,0.001,0.7,0.8],[0.001,0.999,0.7,0.8]))