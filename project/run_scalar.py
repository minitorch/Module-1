"""
Be sure you have minitorch installed in you Virtual Env.
>>> pip install -Ue .
"""
import minitorch
import matplotlib.pyplot as plt
import random
import datasets

PTS = 50
DATASET = datasets.Simple(PTS, vis=True)
HIDDEN = 2
RATE = 0.5


class Network(minitorch.Module):
    def __init__(self):
        super().__init__()

        # Submodules
        self.layer1 = Linear(2, HIDDEN)
        self.layer2 = Linear(HIDDEN, HIDDEN)
        self.layer3 = Linear(HIDDEN, 1)

    def forward(self, x):
        h = [h.relu() for h in self.layer1.forward(x)]
        h = [h.relu() for h in self.layer2.forward(h)]
        return self.layer3.forward(h)[0].sigmoid()


class Linear(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = []
        self.bias = []
        for i in range(in_size):
            self.weights.append([])
            for j in range(out_size):
                self.weights[i].append(
                    self.add_parameter(
                        f"weight_{i}_{j}", minitorch.Scalar(2 * (random.random() - 0.5))
                    )
                )
        for j in range(out_size):
            self.bias.append(
                self.add_parameter(
                    f"bias_{j}", minitorch.Scalar(2 * (random.random() - 0.5))
                )
            )

    def forward(self, inputs):
        y = [b.value for b in self.bias]
        for i, x in enumerate(inputs):
            for j in range(len(y)):
                y[j] = y[j] + x * self.weights[i][j].value
        return y


model = Network()
data = DATASET
losses = []
for epoch in range(500):
    total_loss = 0.0
    correct = 0

    # Forward
    for i in range(data.N):
        x_1, x_2 = data.X[i]
        y = data.y[i]
        x_1 = minitorch.Scalar(x_1)
        x_2 = minitorch.Scalar(x_2)
        out = model.forward((x_1, x_2))

        if y == 1:
            prob = out
            correct += 1 if out.data > 0.5 else 0
        else:
            prob = -out + 1.0
            correct += 1 if out.data < 0.5 else 0

        loss = -prob.log()
        loss.backward()
        total_loss += loss.data

    # Update
    losses.append(total_loss)
    for p in model.parameters():
        if p.value.derivative is not None:
            p.update(
                minitorch.Scalar(p.value.data - RATE * (p.value.derivative / data.N))
            )

    # Logging
    if epoch % 10 == 0:
        print("Epoch ", epoch, " loss ", total_loss, "correct", correct)
        im = f"Epoch: {epoch}"
        data.graph(
            im,
            lambda x: model.forward(
                (minitorch.Scalar(x[0]), minitorch.Scalar(x[1]))
            ).data,
        )
        plt.plot(losses, c="blue")
        data.vis.matplot(plt, win="loss")
