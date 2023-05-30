import numpy as np
import torch.nn as nn
import torch
from einops import rearrange
import torch.nn.functional as F


class SwishActivation(nn.Module):
    def __init__(self):
        # tanh is better than swish
        super(SwishActivation, self).__init__()

    def forward(self, x):
        return torch.sigmoid(x) * x


class Attention(nn.Module):
    def __init__(
        self,
        input_dim,  # input/output dimension
        output_dim,
        heads=8,
        dim_head=16,
    ):
        super().__init__()
        inner_dim = dim_head * heads  # 8 * 16 = 128
        self.heads = heads
        self.scale = dim_head ** -0.5  # 1/ 4
        self.to_qkv = nn.Linear(input_dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, output_dim)

    def forward(self, x):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1) # N x i -> N x 3heads*dim_head -> N x heads*dim_head
        # N x head x dim_head
        q, k, v = map(lambda t: rearrange(t, 'b (h d) -> b h d', h=h), (q, k, v))
        sim = torch.einsum('b h i, b h j -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)  # key softmax
        out = torch.einsum('b h i j, b h j -> b h i', attn, v)
        out = rearrange(out, 'b h i -> b (h i)', h=h) # N x heads*dim_head
        out = self.to_out(out)
        return out


class Block(nn.Module):
    def __init__(self, i_dim, o_dim, use_attention=True):
        super(Block, self).__init__()
        # LBRD
        # if use_attention:
        #     https://github.com/somepago/saint/blob/main/models/model.py
        self.seq_block = nn.Sequential(
            Attention(input_dim=i_dim, output_dim=o_dim),
            nn.BatchNorm1d(o_dim),
            # layernorm + GeLU效果差于BN + Tanh
            # nn.ReLU(inplace=True), # the function is C0
            # nn.ELU(),         # the function is C1
            nn.Tanh(),  # the function is C2
            nn.Dropout() # default p=0.5
        )
        ## origin
        # else:
        #     self.seq_block = nn.Sequential(
        #         nn.Linear(i_dim, o_dim),
        #         nn.BatchNorm1d(o_dim),
        #         # nn.ReLU(inplace=True), # the function is C0
        #         # nn.ELU(),         # the function is C1
        #         nn.Tanh(),          # the function is C2
        #         nn.Dropout()
        #     )

    def forward(self, x):
        return self.seq_block(x)


class DenseBlock(nn.Module):
    def __init__(self, input_dim=20, first_dim=30, second_dim=20, encoding_dim=32):
        super(DenseBlock, self).__init__()
        self.seq_1 = nn.Sequential(
            nn.Linear(input_dim, first_dim),
            # nn.ReLU()
            # nn.ELU()
            nn.Tanh()
        )
        self.deep_block_1 = Block(first_dim, second_dim)
        self.deep_block_2 = Block(second_dim, 32)
        self.deep_block_3 = nn.Sequential(
            nn.BatchNorm1d(32),
            nn.Dropout(),
            nn.Linear(32, encoding_dim),
        )
        self.wide_block_1 = nn.Linear(first_dim, encoding_dim)

    def forward(self, x):
        o = self.seq_1(x)
        o1 = self.wide_block_1(o)
        o2 = self.deep_block_3(self.deep_block_2(self.deep_block_1(o)))
        return o1 + o2


class Net(nn.Module):
    def __init__(self, input_dim=20, first_dim=30, second_dim=20, encoding_dim=32, layer_num=2):
        super(Net, self).__init__()
        self.layer_num = layer_num
        self.input_dim = input_dim
        self.first_dim = first_dim
        self.second_dim = second_dim
        self.encoding_dim = encoding_dim
        self.layers = nn.ModuleList(
            [DenseBlock(input_dim=self.input_dim + encoding_dim * i,
                        first_dim=self.first_dim,
                        second_dim=self.second_dim,
                        encoding_dim=self.encoding_dim)
            for i in range(self.layer_num)])

    def forward(self, x):
        input_ = x
        for i in range(len(self.layers)):
            out = self.layers[i](input_)
            input_ = torch.cat((input_, out), dim=1)
        return out


class MultiBranchModel(nn.Module):
    def __init__(self, input_dim_lst, first_dim=32, second_dim=64, encoding_dim=16, layer_num=2):
        """
            Multi Branch Model
        """
        super(MultiBranchModel, self).__init__()
        self.partition_lst = [0]
        self.model_lst = nn.ModuleList([Net(input_dim, first_dim, second_dim, encoding_dim,
                                            layer_num=layer_num) for input_dim in input_dim_lst])
        for val in input_dim_lst:
            self.partition_lst.append(self.partition_lst[-1] + val)

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(encoding_dim),
            # nn.Dropout(),
            nn.Linear(encoding_dim, 16),
            nn.Linear(16, 1)
        )

    def forward(self, x, istrain=True):
        res = torch.cat([model(x[:, self.partition_lst[i]:self.partition_lst[i + 1]])
                         for i, model in enumerate(self.model_lst)], dim=1)
        # return torch.mean(res, dim=1, keepdim=True)
        if istrain:
            similarity_matrix, arg_order = cal_similarity_matrix(res)  # 计算特征相似性矩阵, bsxbs
            return similarity_matrix, arg_order, self.classifier(res)
        else:
            return self.classifier(res)


def cal_similarity_matrix(data):
    '''
    相似性矩阵：bxb，s(i, j)表示i与j的相似度，每一行按照从大到小排序
    '''
    b, d = data.shape
    sim_matrix, arg_order = [], []
    for i in range(b):
        sim_matrix_i = []
        for j in range(b):
            if i == j:
                sim_matrix_i.append(0.0)
            else:
                vector1, vector2 = data[i].cpu().detach().numpy(), data[j].cpu().detach().numpy()
                num = float(np.dot(vector1, vector2))
                denom = np.linalg.norm(vector1) * np.linalg.norm(vector2)
                sim_matrix_i.append(0.5 + 0.5 * (num / denom) if denom != 0 else 0)
        order = np.argsort(sim_matrix_i)[::-1].tolist()
        arg_order.append(order[:])
        sim_matrix.append(sim_matrix_i[:])
    return sim_matrix, arg_order


if __name__ == "__main__":
    a = torch.rand(5, 25)
    net = MultiBranchModel(input_dim_lst=[12, 13], layer_num=2)
    model_dict = net.state_dict()
    # for k, v in model_dict.items():  # 查看自己网络参数各层名称、数值
    #     print(k)  # 输出网络参数名字
    print(net(a).shape)

