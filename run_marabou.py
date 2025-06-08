from maraboupy import Marabou
import numpy as np

options = Marabou.createOptions(verbosity=0)

filename = "./fmnist_mlp.onnx"
network = Marabou.read_onnx(filename, inputNames=["input"], outputNames=["output"])

inputVars = [int(idx) for idx in network.inputVars[0].flatten()]
outputVars = [int(idx) for idx in network.outputVars[0].flatten()]

for idx in inputVars:
    network.setLowerBound(idx, -1.0)
    network.setUpperBound(idx, 1.0)

network.setLowerBound(outputVars[0], 0.0)
network.setUpperBound(outputVars[0], 10.0)

options = Marabou.createOptions(verbosity=0)
result = network.solve(options=options)
print(result)

import numpy as np
import matplotlib.pyplot as plt

solution_dict = result[1]
inputVars = list(range(784))

input_vector = np.array([solution_dict[idx] for idx in inputVars])
print("input_vector shape:", input_vector.shape)

image = input_vector.reshape(28, 28)
image_vis = (image + 1) / 2.0

plt.imsave('marabou_sat_input.png', image_vis, cmap='gray')

import matplotlib.pyplot as plt
outputs = [0.0, -6.095, -1.240, -2.612, -5.468, -2.512, -0.197, -4.438, -5.336, -4.870]
plt.figure(figsize=(6,4))
plt.bar(range(10), outputs)
plt.xlabel('Class')
plt.ylabel('Output (Logit)')
plt.title('Output vector for Marabou-generated input')
plt.savefig('marabou_output_vector.png')
plt.close()
