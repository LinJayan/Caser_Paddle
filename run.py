import argparse
from time import time

import paddle.optimizer as optim
import paddle.nn.functional as F

from model.caser import Caser
# from interactions import Interactions
from utils.utils import *
from utils.interactions import Interactions


import paddle


class Recommender(object):
    """
    Contains attributes and methods that needed to train a sequential
    recommendation model. Models are trained by many tuples of
    (users, sequences, targets, negatives) and negatives are from negative
    sampling: for any known tuple of (user, sequence, targets), one or more
    items are randomly sampled to act as negatives.


    Parameters
    ----------

    n_iter: int,
        Number of iterations to run.
    batch_size: int,
        Minibatch size.
    l2: float,
        L2 loss penalty, also known as the 'lambda' of l2 regularization.
    neg_samples: int,
        Number of negative samples to generate for each targets.
        If targets=3 and neg_samples=3, then it will sample 9 negatives.
    learning_rate: float,
        Initial learning rate.
    use_cuda: boolean,
        Run the model on a GPU or CPU.
    model_args: args,
        Model-related arguments, like latent dimensions.
    """

    def __init__(self,
                 n_iter=None,
                 batch_size=None,
                 l2=None,
                 neg_samples=None,
                 learning_rate=None,
                 use_cuda=False,
                 model_args=None):

        # model related
        self._num_items = None
        self._num_users = None
        self._net = None
        self.model_args = model_args

        # learning related
        self._batch_size = batch_size
        self._n_iter = n_iter
        self._learning_rate = learning_rate
        self._l2 = l2
        self._neg_samples = neg_samples
        self._device = paddle.device.get_device()

        # rank evaluation related
        self.test_sequence = None
        self._candidate = dict()

    @property
    def _initialized(self):
        return self._net is not None

    def _initialize(self, interactions):
        device = paddle.set_device(self._device ) # 'gpu' or 'gpu'
        self._num_items = interactions.num_items
        self._num_users = interactions.num_users

        self.test_sequence = interactions.test_sequences

        self._net = Caser(self._num_users,
                          self._num_items,
                          self.model_args)

        self._optimizer = optim.Adam(learning_rate=self._learning_rate,
                                     parameters=self._net.parameters(),
                                     weight_decay=self._l2,
                                     )


    def fit(self, train, test, verbose=False):
        """
        The general training loop to fit the model

        Parameters
        ----------

        train: :class:`spotlight.interactions.Interactions`
            training instances, also contains test sequences
        test: :class:`spotlight.interactions.Interactions`
            only contains targets for test sequences
        verbose: bool, optional
            print the logs
        """
        device = paddle.device.set_device(self._device)

        # convert to sequences, targets and users
        sequences_np = train.sequences.sequences
        targets_np = train.sequences.targets
        users_np = train.sequences.user_ids.reshape(-1, 1)

        L, T = train.sequences.L, train.sequences.T

        n_train = sequences_np.shape[0]

        output_str = 'total training instances: %d' % n_train
        print(output_str)

        if not self._initialized:
            self._initialize(train)

        # print('---',self.model_args)
        if self.model_args.mode == 'train':
            start_epoch = 0

            for epoch_num in range(start_epoch, self._n_iter):

                t1 = time()

                # set model to training mode
                self._net.train()

                users_np, sequences_np, targets_np = shuffle(users_np,
                                                            sequences_np,
                                                            targets_np)

                negatives_np = self._generate_negative_samples(users_np, train, n=self._neg_samples)

                # convert numpy arrays to Paddle tensors and move it to the corresponding devices
                users = paddle.to_tensor(users_np,stop_gradient=False)
                sequences = paddle.to_tensor(sequences_np,stop_gradient=False)
                targets = paddle.to_tensor(targets_np,stop_gradient=False)
                negatives = paddle.to_tensor(negatives_np,stop_gradient=False)


                epoch_loss = 0.0

                for (minibatch_num,
                    (batch_users,
                    batch_sequences,
                    batch_targets,
                    batch_negatives)) in enumerate(minibatch(users,
                                                            sequences,
                                                            targets,
                                                            negatives,
                                                            batch_size=self._batch_size)):

                    items_to_predict = paddle.concat([batch_targets, batch_negatives], axis=1)

                    items_prediction = self._net(batch_sequences,batch_users,items_to_predict)

                    (targets_prediction, negatives_prediction) = paddle.split(items_prediction,
                                                        [batch_targets.shape[1],
                                                        batch_negatives.shape[1]], axis=1)

                    # compute the binary cross-entropy loss
                    positive_loss = -paddle.mean(
                        paddle.log(F.sigmoid(targets_prediction)))
                    negative_loss = -paddle.mean(
                        paddle.log(1 - F.sigmoid(negatives_prediction)))
                    loss = positive_loss + negative_loss

                    epoch_loss += loss.item()

                    loss.backward()
                    self._optimizer.step()
                    # 梯度清零
                    self._optimizer.clear_grad()

                epoch_loss /= minibatch_num + 1

                t2 = time()
                if verbose and (epoch_num + 1) % 10 == 0:
                    precision, recall, mean_aps = evaluate_ranking(self, test, train, k=[1, 5, 10])
                    output_str1 = "Epoch %d [%.1f s]\tloss=%.4f [%.1f s]" % (epoch_num + 1,
                                                                        t2 - t1,
                                                                        epoch_loss,
                                                                        time() - t2)
                    output_str2 = "MAP=%.4f, " \
                             "prec@1=%.4f, prec@5=%.4f, prec@10=%.4f, " \
                             "recall@1=%.4f, recall@5=%.4f, recall@10=%.4f, [%.1f s]" % (mean_aps,
                                                                                         np.mean(precision[0]),
                                                                                         np.mean(precision[1]),
                                                                                         np.mean(precision[2]),
                                                                                         np.mean(recall[0]),
                                                                                         np.mean(recall[1]),
                                                                                         np.mean(recall[2]),
                                                                                         time() - t2)
                    print(output_str1)
                    print('Val:',output_str2)

                else:
                    output_str = "Epoch %d [%.1f s]\tloss=%.4f [%.1f s]" % (epoch_num + 1,
                                                                        t2 - t1,
                                                                        epoch_loss,
                                                                        time() - t2)
                    print(output_str)
                
            ##保存模型，会生成**.pdiparams、.pdopt两个模型文件
            # save
            paddle.save(self._net.state_dict(), "./checkpoint/caser.pdparams")
            paddle.save(self._optimizer.state_dict(), "./checkpoint/adam.pdopt")
        else:
            # load
            t3 = time()
            Layer_state_dict = paddle.load("./checkpoint/caser.pdparams")
            opt_state_dict = paddle.load("./checkpoint/adam.pdopt")

            self._net.set_state_dict(Layer_state_dict)
            self._optimizer.set_state_dict(opt_state_dict)

            precision, recall, mean_aps = evaluate_ranking(self, test, train, k=[1, 5, 10])
            output_str = "MAP=%.4f, " \
                                "prec@1=%.4f, prec@5=%.4f, prec@10=%.4f, " \
                                "recall@1=%.4f, recall@5=%.4f, recall@10=%.4f, [%.1f s]" % (mean_aps,
                                                                                            np.mean(precision[0]),
                                                                                            np.mean(precision[1]),
                                                                                            np.mean(precision[2]),
                                                                                            np.mean(recall[0]),
                                                                                            np.mean(recall[1]),
                                                                                            np.mean(recall[2]),
                                                                                            time() - t3)
            print('【Test reporting】:')
            print(output_str)

    

    def _generate_negative_samples(self, users, interactions, n):
        """
        Sample negative from a candidate set of each user. The
        candidate set of each user is defined by:
        {All Items} \ {Items Rated by User}

        Parameters
        ----------

        users: array of np.int64
            sequence users
        interactions: :class:`spotlight.interactions.Interactions`
            training instances, used for generate candidates
        n: int
            total number of negatives to sample for each sequence
        """

        users_ = users.squeeze()
        negative_samples = np.zeros((users_.shape[0], n), np.int64)
        if not self._candidate:
            all_items = np.arange(interactions.num_items - 1) + 1  # 0 for padding
            train = interactions.tocsr()
            for user, row in enumerate(train):
                self._candidate[user] = list(set(all_items) - set(row.indices))

        for i, u in enumerate(users_):
            for j in range(n):
                x = self._candidate[u]
                negative_samples[i, j] = x[
                    np.random.randint(len(x))]

        return negative_samples

    def predict(self, user_id, item_ids=None):
        """
        Make predictions for evaluation: given a user id, it will
        first retrieve the test sequence associated with that user
        and compute the recommendation scores for items.

        Parameters
        ----------

        user_id: int
           users id for which prediction scores needed.
        item_ids: array, optional
            Array containing the item ids for which prediction scores
            are desired. If not supplied, predictions for all items
            will be computed.
        """

        if self.test_sequence is None:
            raise ValueError('Missing test sequences, cannot make predictions')

        # set model to evaluation model
        self._net.eval()
        with paddle.no_grad():
            sequences_np = self.test_sequence.sequences[user_id, :]
            sequences_np = np.atleast_2d(sequences_np)

            if item_ids is None:
                item_ids = np.arange(self._num_items).reshape(-1, 1)

            sequences = paddle.to_tensor(sequences_np,stop_gradient=True)
            items = paddle.to_tensor(item_ids,stop_gradient=True)
            user = paddle.to_tensor(np.array([[user_id]]),stop_gradient=True)

            out = self._net(sequences,
                                user,
                                items,
                                for_pred=True)

        return out.cpu().numpy().flatten()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data arguments
    parser.add_argument('--train_root', type=str, default='datasets/ml1m/test/train.txt')
    parser.add_argument('--test_root', type=str, default='datasets/ml1m/test/test.txt')
    parser.add_argument('--L', type=int, default=5)
    parser.add_argument('--T', type=int, default=3)
    # train arguments
    parser.add_argument('--n_iter', type=int, default=30) #
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--batch_size', type=int, default=512)  # 512
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--l2', type=float, default=1e-6)
    parser.add_argument('--neg_samples', type=int, default=3)
    parser.add_argument('--use_cuda', type=str2bool, default=True)
    parser.add_argument('--mode', type=str,default='train')

    # model arguments (ml1m)
    # parser.add_argument('--d', type=int, default=50)
    # parser.add_argument('--nv', type=int, default=4)
    # parser.add_argument('--nh', type=int, default=16)
    # parser.add_argument('--drop', type=float, default=0.5)
    # parser.add_argument('--ac_conv', type=str, default='relu')
    # parser.add_argument('--ac_fc', type=str, default='relu')

   # -----------------------------------------------------------
   # model arguments (gowalla)
    parser.add_argument('--d', type=int, default=100)
    parser.add_argument('--nv', type=int, default=2)
    parser.add_argument('--nh', type=int, default=16)
    parser.add_argument('--drop', type=float, default=0.5)
    parser.add_argument('--ac_conv', type=str, default='iden')
    parser.add_argument('--ac_fc', type=str, default='sigm')

    config = parser.parse_args()

    # set seed
    paddle.seed(config.seed)


    # load dataset
    train = Interactions(config.train_root)
    # transform triplets to sequence representation
    train.to_sequence(config.L, config.T)

    test = Interactions(config.test_root,
                        user_map=train.user_map,
                        item_map=train.item_map)

    print(config)
    # fit model
    model = Recommender(n_iter=config.n_iter,
                        batch_size=config.batch_size,
                        learning_rate=config.learning_rate,
                        l2=config.l2,
                        neg_samples=config.neg_samples,
                        model_args=config,
                        use_cuda=config.use_cuda)

    model.fit(train, test, verbose=True)
