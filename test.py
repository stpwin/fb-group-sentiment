items = {"a": True, "b": False}

b = [v for k, v in items.items() if v == True]

print(b)
