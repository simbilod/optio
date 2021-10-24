import subprocess

processes = []
for index in range(1, 6):
    command = "python compute-serial" + str(index) + ".py"
    process = subprocess.Popen(command, shell=True)
    processes.append(process)
# Collect statuses
output = [p.wait() for p in processes]
print(output)
