import os
path=r''
def startRename():
    files=os.listdir(path)
    count=1
    for file in files:
        os.rename(f"{path}\{file}",f"{path}\{count}.wav")
        count+=1
startRename()
