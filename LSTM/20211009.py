import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib

# Non-interactive backend, you can't call plt.show() to see the figure interactively
# matplotlib.use('Agg') must be placed before import matplotlib.pyplot
matplotlib.use('Agg')
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def generateSineWave():
    np.random.seed(2)
    T = 20
    L = 1000
    N = 100
    x = np.empty((N, L), 'int64')  # the dataset has 100 items and each item's length is 1000
    x[:] = np.array(range(L)) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)
    data = np.sin(x / 1.0 / T).astype('float64')
    torch.save(data, open('traindata.pt', 'wb'))


class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm1 = nn.LSTMCell(1, 51)
        self.lstm2 = nn.LSTMCell(51, 51)
        self.linear = nn.Linear(51, 1)

    def forward(self, input, future=0):
        outputs = []
        h_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        c_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        h_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)
        c_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)

        h_t = h_t.to(device)
        c_t = c_t.to(device)
        h_t2 = h_t2.to(device)
        c_t2 = c_t2.to(device)

        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)  # output.shape:[batch,1]
            outputs += [output]  # outputs.shape:[[batch,1],...[batch,1]], list composed of n [batch,1],
        for i in range(future):  # if we should predict the future
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)  # output.shape:[batch,1]
            outputs += [output]  # outputs.shape:[[batch,1],...[batch,1]], list composed of n [batch,1],
        outputs = torch.stack(outputs, 1).squeeze(2)  # shape after stack:[batch, n, 1], shape after squeeze: [batch,n]
        return outputs


if __name__ == '__main__':
    # 1. generate sine wave data
    generateSineWave()
    # set random seed to 0
    np.random.seed(0)
    torch.manual_seed(0)
    # load data and make training set
    data = torch.load('traindata.pt')
    input = torch.from_numpy(data[3:, :-1])
    target = torch.from_numpy(data[3:, 1:])
    test_input = torch.from_numpy(data[:3, :-1])
    test_target = torch.from_numpy(data[:3, 1:])
    input = input.to(device)
    target = target.to(device)
    test_input = test_input.to(device)
    test_target = test_target.to(device)
    # 2. build the model
    seq = Sequence()
    seq.double()
    print(seq)
    # move to cuda
    # if torch.cuda.device_count()>1:
    #     seq = nn.DataParallel(seq)
    seq = seq.to(device)

    # 3 loss function
    criterion = nn.MSELoss()
    # 4 use LBFGS as optimizer since we can load the whole data to train
    optimizer = optim.LBFGS(seq.parameters(), lr=0.8)
    # 5 begin to train
    for i in range(1):
        print('STEP: ', i)


        def closure():
            # forward
            out = seq(input)
            loss = criterion(out, target)
            print('loss:', loss.item())
            # backward
            optimizer.zero_grad()
            loss.backward()
            return loss


        optimizer.step(closure)
        # begin to predict, no need to track gradient here
        with torch.no_grad():
            future = 1000
            pred = seq(test_input, future=future)
            loss = criterion(pred[:, :-future], test_target)
            print('test loss:', loss.item())

            y = pred.detach().cpu()
            y = y.numpy()
        # draw the result
        plt.figure(figsize=(30, 10))
        plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
        plt.xlabel('x', fontsize=20)
        plt.ylabel('y', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)


        def draw(yi, color):
            plt.plot(np.arange(input.size(1)), yi[:input.size(1)], color, linewidth=2.0)
            plt.plot(np.arange(input.size(1), input.size(1) + future), yi[input.size(1):], color + ':', linewidth=2.0)


        draw(y[0], 'r')
        draw(y[1], 'g')
        draw(y[2], 'b')
        plt.savefig('predict%d.pdf' % i)
        plt.close()
