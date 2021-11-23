with open("sys_out.txt", "r") as f:
    string = f.read()
    print(string)
    for i in range(len(string)):
        print(string[i])
