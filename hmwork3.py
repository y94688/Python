# 随机数据生成函数
import random
import string


def dataSampling(**kwargs):
    result = {}

    for key, value in kwargs.items():
        if value["type"] == "int":
            result[key] = [random.randint(value["start"], value["end"]) for _ in range(value["num"])]
        elif value["type"] == "float":
            result[key] = [random.uniform(value["start"], value["end"]) for _ in range(value["num"])]
        elif value["type"] == "str":
            result[key] = ["".join(random.choices(value["candidate"], k=value["length"])) for _ in range(value["num"])]
        else:
            result[key] = []

    return result


# 机器学习模型类工厂
class MLFactory:
    @staticmethod
    def create_model(method):
        if method == "SVM":
            return SVMModel()
        elif method == "RF":
            return RFModel()
        elif method == "CNN":
            return CNNModel()
        elif method == "RNN":
            return RNNModel()
        else:
            raise ValueError("Invalid method")


# 评估指标类工厂
class MetricFactory:
    @staticmethod
    def create_metric(name):
        if name == "ACC":
            return Accuracy()
        elif name == "MCC":
            return MatthewsCorrelationCoefficient()
        elif name == "F1":
            return F1Score()
        elif name == "RECALL":
            return Recall()
        else:
            raise ValueError("Invalid metric")


# SVM模型类
class SVMModel:
    def __init__(self):
        print("SVM model created")

    def fit(self, X_train, y_train):
        print("SVM fit")

    def predict(self, X_test):
        print("SVM predict")


# 随机树森林模型类
class RFModel:
    def __init__(self):
        print("RF model created")

    def fit(self, X_train, y_train):
        print("RF fit")

    def predict(self, X_test):
        print("RF predict")


# 卷积神经网络模型类
class CNNModel:
    def __init__(self):
        print("CNN model created")

    def fit(self, X_train, y_train):
        print("CNN fit")

    def predict(self, X_test):
        print("CNN predict")


# 循环神经网络模型类
class RNNModel:
    def __init__(self):
        print("RNN model created")

    def fit(self, X_train, y_train):
        print("RNN fit")

    def predict(self, X_test):
        print("RNN predict")


# 评估指标类
class Metric:
    def __init__(self):
        self.name = ""

    def __call__(self, y_true, y_pred):
        pass


# 准确率评估指标类
class Accuracy(Metric):
    def __init__(self):
        super().__init__()
        self.name = "ACC"

    def __call__(self, y_true, y_pred):
        print("Accuracy calculated")


# 马修斯相关系数评估指标类
class MatthewsCorrelationCoefficient(Metric):
    def __init__(self):
        super().__init__()
        self.name = "MCC"

    def __call__(self, y_true, y_pred):
        print("Matthews correlation coefficient calculated")


# F1分数评估指标类
class F1Score(Metric):
    def __init__(self):
        super().__init__()
        self.name = "F1"

    def __call__(self, y_true, y_pred):
        print("F1 score calculated")


# 召回率评估指标类
class Recall(Metric):
    def __init__(self):
        super().__init__()
        self.name = "RECALL"

    def __call__(self, y_true, y_pred):
        print("Recall calculated")


# 测试代码
if __name__ == '__main__':
    # 生成随机数据
    data = dataSampling(numbers={"type": "int", "start": 1, "end": 100, "num": 10},
                        percentages={"type": "float", "start": 1, "end": 10, "num": 5},
                        names={"type": "str", "candidate": "abcdefghijklmnopqrstuvwxyz", "length": 5, "num": 3})
    print(data)