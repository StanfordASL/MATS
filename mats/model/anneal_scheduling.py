import warnings
import torch.optim as optim
from model.model_utils import rsetattr, rgetattr, CustomLR


def create_new_scheduler(obj, name, annealer, annealer_kws, creation_condition=True):
    value_scheduler = None
    rsetattr(obj, name + '_scheduler', value_scheduler)
    if creation_condition:
        annealer_kws['device'] = obj.device
        value_annealer = annealer(annealer_kws)
        rsetattr(obj, name + '_annealer', value_annealer)

        # This is the value that we'll update on each call of
        # step_annealers().
        rsetattr(obj, name, value_annealer(0).clone().detach())
        dummy_optimizer = optim.Optimizer([rgetattr(obj, name)], {'lr': value_annealer(0).clone().detach()})
        rsetattr(obj, name + '_optimizer', dummy_optimizer)

        value_scheduler = CustomLR(dummy_optimizer,
                                   value_annealer)
        rsetattr(obj, name + '_scheduler', value_scheduler)

    obj.schedulers.append(value_scheduler)
    obj.annealed_vars.append(name)


def step_annealers(obj):
    # This should manage all of the step-wise changed
    # parameters automatically.
    for idx, annealed_var in enumerate(obj.annealed_vars):
        if rgetattr(obj, annealed_var + '_scheduler') is not None:
            # First we step the scheduler.
            with warnings.catch_warnings():  # We use a dummy optimizer: Warning because no .step() was called on it
                warnings.simplefilter("ignore")
                rgetattr(obj, annealed_var + '_scheduler').step()

            # Then we set the annealed vars' value.
            rsetattr(obj, annealed_var, rgetattr(obj, annealed_var + '_optimizer').param_groups[0]['lr'])

    obj.summarize_annealers()


def summarize_annealers(obj, prefix):
    if obj.log_writer is not None:
        for annealed_var in obj.annealed_vars:
            if rgetattr(obj, annealed_var) is not None:
                obj.log_writer.add_scalar('%s/%s' % (prefix, annealed_var.replace('.', '/')),
                                          rgetattr(obj, annealed_var), obj.curr_iter)
