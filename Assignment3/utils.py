def promptMenu(title,options):
    num2option = list(options.keys())
    print(title)
    for i,k in enumerate(options):
        print(f'({i}) - {k}')
    while True:
        ans = input('\nPlease enter one of the numeric keys: ')
        try:
            i = int(ans)
            key = num2option[i]
            return options[key]
        except:
            print(f'Invalid choice: {ans}.\nTry again')

def promptIntegerFromRange(title,valid_options):
    print(title)
    while True:
        ans = input('\nPlease enter a number: ')
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
        ans = input('\nPlease enter a positive number: ')
        try:
            i = int(ans)
            if i > 0:
                return i
            print(f'Invalid choice: {ans}, must be a positive integer\nTry again')
        except:
            print(f'Invalid choice: {ans}, must be an integer\nTry again')

def promptPath(valid_elements):
    print('Enter a path for inference')
    while True:
        ans = input('\nPlease enter a sequnce of nodes seperated by commas: ')
        try:
            path = [int(n) for n in ans.split(',')]
            edges = set()
            for i in range(len(path)-1):
                node1 = path[i]
                node2 = path[i+1]
                n1,n2 = (node1,node2) if node1 < node2 else (node2,node1)
                edges.add((n1,n2))
            if edges.issubset(valid_elements):
                return edges
            print(f'Invalid choice: {ans}, {edges.difference(valid_elements)} are not valid edges \nTry again')
        except:
            print(f'Invalid choice: {ans}, all elements must be integers\nTry again')

def findAllSimplePaths(n1,n2,nodes,edges):
    paths = []
    if n1 == n2:
        return [[n1]]
    source = n1 if n1 < n2 else n2
    target = n2 if n1 < n2 else n1
    all_neighbors = [e[1] for e in edges if e[0] == source]
    for n in all_neighbors:
        for path_suffix in findAllSimplePaths(n,target,nodes,edges):
            paths.append([source,*path_suffix])
    return paths