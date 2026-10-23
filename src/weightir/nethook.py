# Adapted from github.com/kmeng01/rome (MIT License, Copyright (c) 2022 Kevin Meng): util/nethook.py
"""Forward hooks on named submodules: read a module's input or output, edit its output, or stop the forward early."""
import contextlib


class StopForward(Exception):
    """Raised right after a module traced with stop=True has run; the Trace swallows it."""


class Trace(contextlib.AbstractContextManager):
    """Hook one module, by dotted name, for the duration of a ``with`` block.

        with Trace(model, "model.layers.3.mlp.down_proj", retain_input=True, stop=True) as tr:
            model(**batch)
        tr.input    # first positional argument of the module's forward, e.g. (batch, seq, in_dim)
        tr.output   # what the module returned, after edit_output(output) when given

    ``edit_output(output)`` returns the output the rest of the model sees. ``stop=True`` raises StopForward right
    after the module, so later layers do not run.
    """

    def __init__(self, model, layer, *, retain_input=False, retain_output=True, edit_output=None, stop=False):
        self.stop = stop
        self.input = self.output = None

        def hook(module, args, output):
            if retain_input:
                self.input = args[0]
            if edit_output is not None:
                output = edit_output(output)
            if retain_output:
                self.output = output
            if stop:
                raise StopForward()
            return output

        # prepend: run before the hooks transformers installs for output_hidden_states, so they see edited outputs
        self.handle = get_module(model, layer).register_forward_hook(hook, prepend=True)

    def __exit__(self, exc_type, exc, tb):
        self.handle.remove()
        return self.stop and exc_type is not None and issubclass(exc_type, StopForward)


def get_module(model, name):
    """The submodule called ``name``; LookupError if there is none."""
    for n, m in model.named_modules():
        if n == name:
            return m
    raise LookupError(name)
