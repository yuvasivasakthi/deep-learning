import pandas as pd
from sklearn.linear_model import Perceptron

if __name__ == "__main__":
      
       data = pd.read_csv('diabetes.csv').values
       x = data[:, 0:8]
       y = data[:, 8]
       model = Perceptron(random_state=1)
       model.fit(x,y)
       print("%0.3f" % model.score(x,y))
