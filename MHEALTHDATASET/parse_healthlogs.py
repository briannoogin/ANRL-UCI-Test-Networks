import sys

with open(sys.argv[2], mode="w") as out:
    with open(sys.argv[1]) as log:
        for line in log:
            log_samples = line.split()
            out.write("".join(num.ljust(12) for num in log_samples))
            out.write("\n");