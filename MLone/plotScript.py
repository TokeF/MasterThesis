import json
import matplotlib.pyplot as plt
from pprint import pprint
	
with open("histFile.json") as json_file:
		s = json.load(json_file)
loss = s["loss"]
print(loss)
print(type(loss))
x = range(1,6)
plt.plot(x,loss)
plt.figure()
plt.plot(x,s["acc"])
plt.show()