class MultipleOptimizer(object):
    def __init__(self, op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()

    def __getitem__(self, idx):
        return self.optimizers[idx]


class MultipleScheduler(object):
    def __init__(self, sc):
        self.schedulers = sc

    def step(self):
        for sc in self.schedulers:
            sc.step()

    def __getitem__(self, idx):
        return self.schedulers[idx]
