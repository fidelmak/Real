from platform import platform
from real import *
print(f"Welcome to the Real interpreter, built with Python, running on: {platform()}\nTIP: Press Control+C to quit the interpreter.\n")
while True:
    try:
        text = input(">>>> ")
    except KeyboardInterrupt:
        break
    res, err = run('<stdin>', text)
    if err: print(err.as_string())
    else: print(res)