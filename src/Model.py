import math
import torch
import torch.nn.functional as F


default_device = "cpu"
context_length = 3


class Layer:
    def __init__(self):
        self.training = True
    
    def eval(self):
        self.training = False
    
    def train(self):
        self.training = True
    
    def parameters(self):
        return []


class Linear(Layer):
    def __init__(self, input_dim, output_dim, gain=1.0, bias=False, d=default_device, scale_factor=1.0):
        super().__init__()
        self.bias = bias

        self.W = torch.randn(input_dim, output_dim, device=d) * gain * scale_factor
    
        if self.bias is True:
            self.b = torch.randn(output_dim, device=d)
        self.out = None
    
    def __call__(self, X):
        self.out = X @ self.W 
        if self.bias is True:
            self.out += self.b
        return self.out
    
    def parameters(self):
        params = [self.W]
        if self.bias is True:
            params.append(self.b)
        return params


class Flatten(Layer):
    def __init__(self, output_dim):
        super().__init__()
        self.output_dim = output_dim
        

    def __call__(self, X):
        self.out = X.view(-1, self.output_dim)
        return self.out
    

class Tanh(Layer):
    def __init__(self):
        super().__init__()

    def __call__(self, X):
        self.out = torch.tanh(X)
        return self.out



class Sequential(Layer):
    def __init__(self, *layers):
        super().__init__()
        self.layers = [l for l in layers]
    
    def __call__(self, X):
        self.out = X
        for l in self.layers:
            self.out = l(self.out)
        return self.out

    def eval(self):
        for l in self.layers:
            l.eval()
    
    def train(self):
        for l in self.layers:
            l.train()

    def parameters(self):
        params = []
        for l in self.layers:
            params.extend(l.parameters())
        return params


class Embedding(Layer):
    def __init__(self, input_dim, output_dim, d=default_device):
        super().__init__()
        self.C = torch.randn(input_dim, output_dim, device=d)

    def __call__(self, X):
        self.out = self.C[X]
        return self.out

    def parameters(self):
        return [self.C]


class BatchNorm1d(Layer):
    def __init__(self, dim, eps= 0.0001, modemtum=0.001, d=default_device):
        super().__init__()

        self.bngain = torch.ones(dim, device=d)
        self.bnbias = torch.zeros(dim, device=d)

        self.bnmean_running = torch.zeros(dim, device=d)
        self.bnstd_running = torch.ones(dim, device=d)

        self.eps = eps
        self.momentum = modemtum


    def __call__(self, X):
        if self.training is True:
            mean_i = X.mean(dim=0, keepdim=True)
            std_i = X.std(dim=0, keepdim=True)

            with torch.no_grad():
                self.bnmean_running = (1-self.momentum) * self.bnmean_running + self.momentum * mean_i
                self.bnstd_running = (1-self.momentum) * self.bnstd_running + self.momentum * std_i
        else:
            mean_i = self.bnmean_running
            std_i = self.bnstd_running

        X_norm = (X - mean_i) / (std_i + self.eps)
        self.out = X_norm * self.bngain + self.bnbias
        return self.out
    
    def parameters(self):
        return [self.bngain, self.bnbias]
    


class Model:
    """
    Created to access each layer individualy
    """
    def __init__(self, vocab_len, feature_dim, n_hidden, context_length, d=default_device):

        input_dim = context_length * feature_dim

        self.emb = Embedding(vocab_len, feature_dim, d=d)
        self.flatten = Flatten(input_dim)
        self.linear_1 = Linear(input_dim, n_hidden, d=d, bias=False, gain=(5/3), scale_factor=(1/math.sqrt(input_dim)))
        self.batchNorm1d = BatchNorm1d(n_hidden, d=d)
        self.tanh = Tanh()
        self.linear_2 = Linear(n_hidden, vocab_len, d=d, gain=1.0, bias=False, scale_factor=0.01)
        
        self.layers = Sequential(
            self.emb,
            self.flatten,
            self.linear_1,
            self.batchNorm1d,
            self.tanh,
            self.linear_2
        )

    def eval(self):
        self.layers.eval()
    
    def train(self):
        self.layers.train()
    
    def __call__(self, X):
        self.out = self.layers(X)
        return self.out
    
    def parameters(self):
        return self.layers.parameters()


@torch.no_grad()
def estimateError(X, Y, model, batch_size=100, loop_count=100, d=default_device):
    model.eval()
    lossi = []
    for i in range(loop_count):
        ix = torch.randint(X.shape[0], size=(batch_size, ), device=d)
        X_i, Y_i = X[ix], Y[ix]

        logits = model(X_i)
        lossi.append(F.cross_entropy(logits, Y_i).item()) 
    model.train()
    return sum(lossi) / len(lossi)


@torch.no_grad()
def generate(model, encode_func, decode_func, start_ch=".", context_length=context_length, count=10, max_char=10, d=default_device):

    retval = []
    model.eval()
    for i in range(count):
        current_context = encode_func([start_ch] * context_length)
        w = []
        while len(w) < max_char:
            probs = F.softmax(model(torch.tensor([current_context], device=d)), dim=-1)
            ix = torch.multinomial(probs, num_samples=1)

            ch = decode_func(ix.squeeze().item())
            if ch == start_ch:
                break
            
            w.append(ch)
            current_context.pop(0)
            current_context.append(ix)

        retval.append("".join(w))
    model.train()
    return retval