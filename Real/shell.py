from real import *
print("Welcome to the Real interpreter, built with Python\nTIP: Press Control+C to quit the interpreter.\n")
while True:
    try:
        text = input("Real> ")
    except KeyboardInterrupt:
        break
    res, err = run('<stdin>', text)
    if err: print(err.as_string())
    else: print(res)