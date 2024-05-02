import torch
from .operations import DEFAULT_HOOKS, DEFAULT_OPERATIONS, neutral_decorator, neutral_hook
from .group import RelatedGroup
from ..utils import remove_all_hooks


class GroupTracer(torch.fx.Interpreter):
    def __init__(
        self,
        model,
        ignored_dims=None
    ):
        gm = self.symbolic_trace(model)
        super().__init__(gm, True)
        self.model = model
        self.groups = None

        self.ignored_dims = ignored_dims

        self.default_operations = DEFAULT_OPERATIONS
        self.default_hooks = DEFAULT_HOOKS

    def symbolic_trace(self, model):
        return torch.fx.symbolic_trace(model)

    def filter_groups(self):
        if self.ignored_dims is None:
            return ...
        
        indices_to_delete = set()

        for i, group in enumerate(self.groups):
            if any(
                p['name'] in self.ignored_dims 
                and 
                p['dim'] in self.ignored_dims[p['name']] 
                for p in group.params
            ):
                indices_to_delete.add(i)

        self.groups = [
            group 
            for i, group in enumerate(self.groups) 
            if i not in indices_to_delete
        ]

    def build_groups(self, *args, initial_env=None, enable_io_processing=True):
        self.groups = []
        self.model.eval()

        for name, param in self.model.named_parameters():
            param.related_groups = [None] * param.ndim
            layer_name, _ = name.rsplit('.', 1)
            layer = self.model.get_submodule(layer_name)
            dims = list(range(param.ndim))

            for dim in dims:
                size = param.shape[dim]
                group = RelatedGroup(size)
                group.append_param(name, param, dim, layer)
                param.related_groups[dim] = group
                self.groups.append(group)

        device = next(iter(self.model.parameters())).device
        args = list(args)

        for i in range(len(args)):
            if type(args[i]) == torch.Tensor:
                args[i] = args[i].to(device)
            else:
                args[i] = torch.rand(args[i]).to(device)

        self.run(*args, initial_env, enable_io_processing)
        remove_all_hooks(self.model)
        self.groups = [group for group in self.groups if len(group.params) > 1]
        
        self.filter_groups()

        def add_parent_groups(group, parents):
            for parent in group.parents:
                if parent not in parents:
                    parents.add(parent)
                add_parent_groups(parent, parents)

        parents = set()

        for group in self.groups:
            add_parent_groups(group, parents)
            group.build_operations_set()

        for parent in parents:
            parent.build_operations_set()

        self.groups.extend(list(parents))

        return self.groups

    def call_function(self, target, args, kwargs):
        if target in self.default_operations:
            out = self.default_operations[target](*args, **kwargs)
        if target is getattr:
            out = super().call_function(target, args, kwargs)
        else:
            out = neutral_decorator(target)(*args, **kwargs)

        return out

    def call_method(self, target, args, kwargs):
        if target in self.default_operations:
            return self.default_operations[target](*args, **kwargs)
        else:
            return super().call_method(target, args, kwargs)

    def call_module(self, target, args, kwargs):
        submod = self.fetch_attr(target)

        if type(submod) in self.default_hooks:
            submod.register_forward_hook(self.default_hooks[type(submod)])
        else:
            submod.register_forward_hook(neutral_hook)

        return submod(*args, **kwargs)
