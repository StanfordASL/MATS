

class Dynamic(object):
    def __init__(self, dt, dyn_limits, device, model_registrar, node_type,
                 batch_size=None, num_components=None):
        self.dt = dt
        self.device = device
        self.dyn_limits = dyn_limits
        self.initial_conditions = None
        self.model_registrar = model_registrar
        self.node_type = node_type
        self.init_constants(batch_size, num_components)

    def set_initial_condition(self, init_con):
        self.initial_conditions = init_con

    def init_constants(self, batch_size, num_components):
        pass

    def integrate_samples(self, s, x):
        raise NotImplementedError

    def integrate_distribution(self, dist, x):
        raise NotImplementedError
