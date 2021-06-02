def promptMenu(title,options):
    num2option = list(options.keys())
    print(title)
    for i,k in enumerate(options):
        print(f'({i}) - {k}')
    while True:
        ans = input('Please enter one of the numeric keys: ')
        try:
            i = int(ans)
            key = num2option[i]
            return options[key]
        except:
            print(f'Invalid choice: {ans}.\nTry again')

def promptIntegerFromRange(title,valid_options):
    print(title)
    while True:
        ans = input('Please enter a number: ')
        try:
            i = int(ans)
            if i in valid_options:
                return i
            else:
                print(f'Invalid choice: {i}, number must be one of: {valid_options}.\nTry again')
        except:
            print(f'Invalid choice: {ans}, must be an integer\nTry again')

def promptIntegerAny(title):
    print(title)
    while True:
        ans = input('Please enter a number: ')
        try:
            i = int(ans)
            return i
        except:
            print(f'Invalid choice: {ans}, must be an integer\nTry again')

def promptIntegerPositive(title):
    print(title)
    while True:
        ans = input('Please enter a positive number: ')
        try:
            i = int(ans)
            if i > 0:
                return i
            print(f'Invalid choice: {ans}, must be a positive integer\nTry again')
        except:
            print(f'Invalid choice: {ans}, must be an integer\nTry again')