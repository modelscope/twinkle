

class InputProcessor:

    def __call__(self, inputs, device_mesh):
        return self.prepare_task_inputs(self.redistribute(inputs))

    def redistribute(self, inputs):
        return inputs

    def prepare_task_inputs(self, inputs):
        return inputs