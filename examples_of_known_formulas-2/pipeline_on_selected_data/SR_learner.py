import numpy as np
from pysr import PySRRegressor
import torch
import sympy


class BaseLearner:
    def __init__(self, save_dir='checkpoints_SR/'):
        self.save_dir = save_dir
        self.model = PySRRegressor(
            niterations=30,
            # select_k_features=6,
            # binary_operators=["*", "+", "-", "/"],
            # unary_operators=["exp", "log", "inv(x) = 1/x"],
            # extra_sympy_mappings={'inv': lambda x: 1 / x},
            # constraints={'pow': (1, 1), 'mult': (3, 3)},

            binary_operators=["+", "-", "*", "/", "^"],
            unary_operators=["square",
                             "exp",
                             "sqrt",
                             "inv(x) = 1/x"],
            extra_sympy_mappings={'inv': lambda x: 1 / x},
            constraints={
                "/": (-1, 9),
                "exp": 9
            },
            # nested_constraints={
            #     "square": {"square": 1, "cube": 1, "exp": 0},
            #     "cube": {"square": 1, "cube": 1, "exp": 0},
            #     "exp": {"square": 1, "cube": 1, "exp": 0},
            # },
            # complexity_of_operators={"/": 2, "exp": 3},
            # complexity_of_constants=2,

            # extra_torch_mappings={sympy.core.numbers.Rational: torch.FloatTensor,
            #                       sympy.core.numbers.Half: torch.FloatTensor},
            model_selection='best',
            loss="loss(x, y) = abs(x - y)",  # Custom loss function (julia syntax)

            # optimizer_algorithm="BFGS",
            # optimizer_iterations=10,
            # optimize_probability=1,
            tempdir=self.save_dir + "formulas/",
            # verbosity=1,
            temp_equation_file=True,  # 是否生成临时文件
            delete_tempfiles=False,  # 是否最后删除临时文件（不能选True）
        )

    def fit(self, x, y):
        self.model.fit(x, y)
        return self.model.sympy()

    def get_prediction(self, x):
        pred_y = self.model.predict(x)
        return pred_y

    def get_prediction_loss(self, x, y):
        pred_y = self.model.predict(x)
        pred_y_tensor = torch.Tensor(pred_y.reshape(-1, 1))
        loss = torch.mean((pred_y_tensor - torch.Tensor(y.reshape(-1, 1))) ** 2)
        return loss

