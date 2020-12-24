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


print(do([0.3,0.84,0.3,0.2],[0.999,0.16,0.3,0.2]))