import paddle
import paddle.nn as nn
import paddle.tensor as tensor
import paddle.nn.functional as F
from paddle.nn import Linear, Layer, Embedding, LayerNorm, Tanh

from utils.utils import activation_getter


class Caser(Layer):
    """
     Parameters
    ----------

    num_users: int,
        Number of users.
    num_items: int,
        Number of items.
    model_args: args,
        Model-related arguments, like latent dimensions.
    """
    
    def __init__(self, num_users, num_items, model_args):
        super(Caser, self).__init__()
        self.args = model_args

        # init args
        L = self.args.L
        dims = self.args.d
        self.n_h = self.args.nh
        self.n_v = self.args.nv
        self.drop_ratio = self.args.drop
        self.ac_conv = activation_getter[self.args.ac_conv]
        self.ac_fc = activation_getter[self.args.ac_fc]

        # user and item embeddings
        self.user_embeddings = nn.Embedding(num_users, dims)
        self.item_embeddings = nn.Embedding(num_items, dims)

        # vertical conv layer
        self.conv_v = nn.Conv2D(1, self.n_v, (L, 1))

        # horizontal conv layer
        lengths = [i + 1 for i in range(L)]
        # self.conv_h = nn.ModuleList([nn.Conv2d(1, self.n_h, (i, dims)) for i in lengths])
        self.conv_h = [nn.Conv2D(1, self.n_h, (i, dims)) for i in lengths]
     

        # fully-connected layer
        self.fc1_dim_v = self.n_v * dims
        self.fc1_dim_h = self.n_h * len(lengths)
        fc1_dim_in = self.fc1_dim_v + self.fc1_dim_h
        # W1, b1 can be encoded with nn.Linear
        self.fc1 = nn.Linear(fc1_dim_in, dims)
        # W2, b2 are encoded with nn.Embedding, as we don't need to compute scores for all items
        self.W2 = nn.Embedding(num_items, dims+dims)
        self.b2 = nn.Embedding(num_items, 1)

        # dropout
        self.dropout = nn.Dropout(self.drop_ratio)

        # weight initialization
        user_emb_weight = paddle.framework.ParamAttr(name="user_embedding_weight",initializer=paddle.nn.initializer.Normal(mean=0.0, std=1.0 / dims))
        item_emb_weight = paddle.framework.ParamAttr(name="item_embedding_weight",initializer=paddle.nn.initializer.Normal(mean=0.0, std=1.0 / dims))
        w2_emb_weight = paddle.framework.ParamAttr(name="w2_embedding_weight",initializer=paddle.nn.initializer.Normal(mean=0.0, std=1.0 / dims))
        self.user_embeddings.weight_attr = user_emb_weight
        self.item_embeddings.weight_attr = item_emb_weight
        self.W2.weight_attr = w2_emb_weight 
        self.b2.weight_attr = paddle.nn.initializer.Constant(value=0.0)     

        self.cache_x = None

    
    def forward(self, seq_var, user_var, item_var, for_pred=False):
        """
        The forward propagation used to get recommendation scores, given
        triplet (user, sequence, targets).

        Parameters
        ----------

        seq_var: torch.FloatTensor with size [batch_size, max_sequence_length]
            a batch of sequence
        user_var: torch.LongTensor with size [batch_size]
            a batch of user
        item_var: torch.LongTensor with size [batch_size]
            a batch of items
        for_pred: boolean, optional
            Train or Prediction. Set to True when evaluation.
        """

        # Embedding Look-up
        item_embs = paddle.unsqueeze(self.item_embeddings(seq_var),axis=1) # use unsqueeze() to get 4-D
        user_emb = paddle.squeeze(self.user_embeddings(user_var),axis=1)

        # Convolutional Layers
        out, out_h, out_v = None, None, None
        # vertical conv layer
        if self.n_v:
            out_v = self.conv_v(item_embs)
            out_v = paddle.reshape(out_v, [-1, self.fc1_dim_v]) # prepare for fully connect

        # horizontal conv layer
        out_hs = list()
        if self.n_h:
            for conv in self.conv_h:
                # conv_out = self.ac_conv(conv(item_embs).squeeze(3))
                conv_out = paddle.squeeze(self.ac_conv(conv(item_embs)),axis=3)
                MaxPool1D = nn.layer.MaxPool1D(kernel_size=conv_out.shape[2], stride=2, padding=0)
                
                # pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
                pool_out = MaxPool1D(conv_out)
                pool_out = paddle.squeeze(pool_out,axis=2)
                out_hs.append(pool_out)
            out_h = paddle.concat(out_hs, axis=1) # prepare for fully connect

        # Fully-connected Layers
        out = paddle.concat([out_v, out_h], axis=1)
        # apply dropout
        out = self.dropout(out)

        # fully-connected layer
        z = self.ac_fc(self.fc1(out))
        x = paddle.concat([z, user_emb], axis=1)

        w2 = self.W2(item_var)
        b2 = self.b2(item_var)

        if for_pred:
            w2 = paddle.squeeze(w2)
            b2 = paddle.squeeze(b2)
            res = paddle.sum((x * w2),axis=1) + b2
        else:
            # res = w2*x + b
            res = paddle.bmm(w2,paddle.unsqueeze(x,axis=2)) + b2
            res = paddle.squeeze(res)

        return res


