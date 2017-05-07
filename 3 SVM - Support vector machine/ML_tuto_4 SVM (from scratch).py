import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')

class Support_Vector_Machine:
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1: 'r', -1: 'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)

    # train
    def fit(self, data):
        # add the input data so we can reference it from other methods
        self.data = data
        # { |w|: [w,b] }
        opt_dict = {}

        # мора да ги испробаме сите трансформации оти кога бараме должина на вектор бараме корен од квадрати,
        # па би добиле исти вредности за + и -, затоа вака
        transforms = [[1,1],
                      [-1,1],
                      [-1,-1],
                      [1,-1]]

        # we need to find the min and max input data values
        # later we use them in the bowl
        all_data = []
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)
        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data = None

        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      # point of expence 
                      self.max_feature_value * 0.001]

        # we don't need to take as small of steps with b as we do w 
        # extremely expensive, but oh well...
        b_range_multiple = 5
        b_multiple = 5
        # we start with this value for the edge of the bawl
        latest_optimum = self.max_feature_value * 10

        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum])
            # we can do this because convex
            optimized = False
            while not optimized:
                for b in np.arange(-1 * self.max_feature_value * b_range_multiple,
                                   self.max_feature_value * b_range_multiple,
                                   step * b_multiple):
                    for transformation in transforms:
                        # we trransform w into w_t
                        w_t = w * transformation
                        found_option = True
                        #weakest link in the SVM fundamentaly, SMO attempts to fix this a bit
                        # Check the constraint yi (xi . w + b) >= 1 for all our data 
                        for i in self.data:
                            for xi in self.data[i]:
                                # i is the class i.e. yi
                                yi = i
                                if not yi*(np.dot(w_t, xi)+b) >= 1:
                                    found_option = False            #I can add a break here
                                    break
                            if not found_option:
                                break

                        # if everything checked out, add our |w_t| and b values in our opt_dictionary             
                        if found_option:
                            # { |w_t| : [ w_t, b ]
                            opt_dict[np.linalg.norm(w_t)] = [w_t, b]
                            
                # once we've finnished with every b and w transformation option...
                # both values in w are identical, we can use w[1] too...
                if w[0] < 0:
                    optimized = True
                    print('Optimized a step.')
                else:
                    # The bowl analogy. Step down the |w| value and try again
                    # w = [5,5]
                    # step = 1
                    # w - step = [4,4]
                    w = w - step

            # sort opt_dict and get the minimal |w| key 
            norms = sorted([n for n in opt_dict])
            opt_choice = opt_dict[norms[0]]
            # bet vector w and b
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            # 
            latest_optimum = opt_choice[0][0] + step * 2
        
        
    def predict(self, features):
        #sign(X . W + b)
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)
        if classification != 0 and self.visualization:
            self.ax.scatter(features[0], features[1], s=200, marker='*', c=self.colors[classification])
        return classification

    def visualize(self):
        [[self.ax.scatter(x[0], x[1], s=100, color=self.colors[i]) for x in data_dict[i]] for i in data_dict]

        # hyperplane  = x . w + b
        # v = x . w + b
        # psv  = 1
        # nsv = -1
        # dec = 0
        def hyperplane(x, w, b, v):
            return (-w[0]*x-b+v) / w[1]

        datarange = (self.min_feature_value * 0.9, self.max_feature_value * 1.1)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]

        # positive support vector hyperplane
        # w . x + b = 1
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min, hyp_x_max], [psv1, psv2], 'k')

        # negative support vector hyperplane
        # w . x + b = -1
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min, hyp_x_max], [nsv1, nsv2], 'k')

        # desecion boundary hyperplane
        # w . x + b = 0
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min, hyp_x_max], [db1, db2], 'y--')

        plt.show()
    

if __name__ == '__main__':

    data_dict = {-1: np.array([[1,7],[2,8],[3,8]]),
                  1: np.array([[5,1],[6,-1],[7,3]])}

    svm = Support_Vector_Machine()
    svm.fit(data=data_dict)

    predict_us = [[0,10], [1,3], [3,4], [3,5], [5,5], [5,6], [6,-5], [5,8]]

    for p in predict_us:
        svm.predict(p)
    
    svm.visualize()








    
