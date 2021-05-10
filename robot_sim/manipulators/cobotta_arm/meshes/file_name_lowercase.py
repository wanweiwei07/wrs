import os

first = os.listdir()
for file in os.listdir():
    # if you do not want to change the name of the .py file too uncomment the next line
    # if not file.endswith(".py") # and indent the next one (of four spaces)
    os.rename(file, file.lower())  # use upper() for the opposite goal

then = os.listdir()
print("Done, all files to lower case")
for file, file2 in zip(first, then):
    print(file, "-", file2)