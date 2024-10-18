import os
import pyRAPL
import json


# sudo chmod -R a+r /sys/class/powercap/intel-rapl

# @measure_energy(domains=[RaplPackageDomain(0)], handler=csv_handler)
# perf_command = "sudo perf stat -e power/energy-cores/"
# os.system(perf_command + " python -m rl_zoo3.train --algo ppo --env mobile-small-central-v0 --n-jobs 8 --num-threads 16 -P -tb Tensorboard-Comparison")

pyRAPL.setup()
meter = pyRAPL.Measurement("Run")
csv_output = pyRAPL.outputs.CSVOutput('result.csv')

meter.begin()

# Program begins here
os.system("python -m rl_zoo3.train --algo ppo --env mobile-small-central-v0 --n-jobs 8 --num-threads 16 -P -tb Tensorboard-Comparison")
# for i in range(1000):
#     print(i)
# meter.end()

meter.export(csv_output)
csv_output.save()
