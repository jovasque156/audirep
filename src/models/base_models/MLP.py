from torch.nn import Module, Linear, ReLU, Sigmoid, Softmax, LeakyReLU, Dropout, Sequential, LayerNorm

# from https://github.com/ahxt/fair_fairness_benchmark/blob/master/src/networks.py#L62C1-L80C35
class MLP(Module):
    def __init__(self, input_size, hidden_layers, output_size, activation_hidden=None, activation_output=None, p_dropout=None):
        super(MLP, self).__init__()
        self.output_size = output_size
        
        layers = []

        if hidden_layers:
            shapes =[input_size]+hidden_layers if isinstance(hidden_layers, list) else [input_size]+[hidden_layers]
        
            # Add hidden layers
            for i, o in zip(shapes[:-1], shapes[1:]):
                m = Linear(i, o)
                layers.append(m)
                if activation_hidden is not None:
                    if activation_hidden=='relu':
                        layers.append(ReLU())
                    if activation_hidden=='sigmoid':
                        layers.append(Sigmoid())
                    if activation_hidden=='softmax':
                        layers.append(Softmax(dim=1))
                    if activation_hidden=='leaky-relu':
                        layers.append(LeakyReLU(0.01))
                if p_dropout is not None:
                    layers.append(Dropout(p_dropout))
                bn = LayerNorm(o)
                layers.append(bn)
        else:
            o = input_size

        # Add output layer
        if not hidden_layers:
            layers.append(LayerNorm(input_size))

        out = Linear(o, self.output_size)
        layers.append(out)
        if activation_output is not None:
            if activation_output=='relu':
                layers.append(ReLU())
            if activation_output=='sigmoid':
                layers.append(Sigmoid())
            if activation_output=='softmax':
                layers.append(Softmax(dim=1))
            if activation_output=='leaky-relu':
                layers.append(LeakyReLU(0.01))

        self.network = Sequential(*layers)

    def forward(self, x):
        return self.network(x)