import ast
import os
import inspect
from pathlib import Path
from typing import Dict, List, Set, Tuple


def generate_processors():
    from twinkle.dataloader import DataLoader
    from twinkle.dataset import Dataset
    from twinkle.hub import HubOperation
    from twinkle.preprocessor import Preprocessor
    from twinkle.processor import InputProcessor
    from twinkle.reward import Reward
    from twinkle.template import Template
    from twinkle.weight_loader import WeightLoader

    # Base classes to search for
    base_classes = {
        'DataLoader': DataLoader,
        'Dataset': Dataset,
        'HubOperation': HubOperation,
        'Preprocessor': Preprocessor,
        'InputProcessor': InputProcessor,
        'Reward': Reward,
        'Template': Template,
        'WeightLoader': WeightLoader,
    }
    
    # Map module names to their paths
    module_mapping = {
        'dataloader': 'dataloader',
        'dataset': 'dataset',
        'hub': 'hub',
        'preprocessor': 'preprocessor',
        'processor': 'processor',
        'reward': 'reward',
        'template': 'template',
        'weight_loader': 'weight_loader',
    }
    
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    src_twinkle_path = project_root / 'src' / 'twinkle'
    src_client_path = project_root / 'src' / 'client'
    
    def find_classes_in_file(file_path: Path) -> List[Tuple[str, str, List[Tuple[str, str]]]]:
        """Find classes that inherit from base classes.
            
        Returns:
            List of tuples (class_name, base_class_name, [(method_name, signature), ...])
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read(), filename=str(file_path))
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return []
            
        def get_method_signature(func_node: ast.FunctionDef) -> str:
            """Extract method signature from AST node."""
            args = []
                
            # Regular arguments
            for i, arg in enumerate(func_node.args.args):
                if arg.arg == 'self':
                    continue
                    
                # Get argument name
                arg_str = arg.arg
                    
                # Get type annotation if available
                if arg.annotation:
                    try:
                        arg_str += f": {ast.unparse(arg.annotation)}"
                    except:
                        pass
                    
                # Get default value if available
                defaults_offset = len(func_node.args.args) - len(func_node.args.defaults)
                if i >= defaults_offset:
                    default_idx = i - defaults_offset
                    try:
                        default_val = ast.unparse(func_node.args.defaults[default_idx])
                        arg_str += f" = {default_val}"
                    except:
                        pass
                    
                args.append(arg_str)
                
            # *args
            if func_node.args.vararg:
                vararg_str = f"*{func_node.args.vararg.arg}"
                if func_node.args.vararg.annotation:
                    try:
                        vararg_str += f": {ast.unparse(func_node.args.vararg.annotation)}"
                    except:
                        pass
                args.append(vararg_str)
                
            # **kwargs
            if func_node.args.kwarg:
                kwarg_str = f"**{func_node.args.kwarg.arg}"
                if func_node.args.kwarg.annotation:
                    try:
                        kwarg_str += f": {ast.unparse(func_node.args.kwarg.annotation)}"
                    except:
                        pass
                args.append(kwarg_str)
                
            return ', '.join(args)
            
        classes_found = []
            
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Check if this class inherits from any base class
                for base in node.bases:
                    base_name = None
                    if isinstance(base, ast.Name):
                        base_name = base.id
                    elif isinstance(base, ast.Attribute):
                        base_name = base.attr
                        
                    if base_name in base_classes:
                        # Extract all methods decorated with @remote_function
                        methods = []
                        for item in node.body:
                            if isinstance(item, ast.FunctionDef):
                                # Check for remote_function decorator
                                has_remote_decorator = False
                                for decorator in item.decorator_list:
                                    if isinstance(decorator, ast.Name) and decorator.id == 'remote_function':
                                        has_remote_decorator = True
                                    elif isinstance(decorator, ast.Call):
                                        if isinstance(decorator.func, ast.Name) and decorator.func.id == 'remote_function':
                                            has_remote_decorator = True
                                        elif isinstance(decorator.func, ast.Attribute) and decorator.func.attr == 'remote_function':
                                            has_remote_decorator = True
                                    
                                if has_remote_decorator and not item.name.startswith('_'):
                                    signature = get_method_signature(item)
                                    methods.append((item.name, signature))
                            
                        classes_found.append((node.name, base_name, methods))
            
        return classes_found

    def scan_module(module_path: Path) -> Dict[str, List[Tuple[str, List[Tuple[str, str]]]]]:
        """Scan a module directory for classes.
            
        Returns:
            Dict mapping base_class_name to list of (class_name, [(method_name, signature), ...])
        """
        result = {}
            
        if not module_path.exists():
            return result
            
        for py_file in module_path.glob('*.py'):
            if py_file.name.startswith('_'):
                continue
                
            classes = find_classes_in_file(py_file)
            for class_name, base_name, methods in classes:
                if base_name not in result:
                    result[base_name] = []
                result[base_name].append((class_name, methods))
            
        return result
        
    def generate_client_class(class_name: str, base_class_name: str, 
                             methods: List[Tuple[str, str]], module_name: str, 
                             processor_type: str) -> str:
        """Generate client wrapper class code.
            
        Args:
            methods: List of tuples (method_name, signature)
        """
        code_lines = [f"""from typing import Any, Optional
import uuid
from client.http import TWINKLE_SERVER_URL
from client.http import http_post, heartbeat_manager
import twinkle


class {class_name}(twinkle.{module_name}.{base_class_name}):
    \"\"\"Client wrapper for {class_name} that calls server HTTP endpoints.\"\"\"
    
    def __init__(self,
                 **kwargs):
        assert TWINKLE_SERVER_URL
        self.server_url = TWINKLE_SERVER_URL
        
        # Create processor instance on server
        response = http_post(
            url=f'{{self.server_url}}/create',
            json_data={{
                'processor_type': '{processor_type}',
                'class_type': '{class_name}',
                **kwargs
            }}
        )
        response.raise_for_status()
        self.processor_id = response.json()
        
        # Register for automatic heartbeat
        heartbeat_manager.register_processor(
            self.processor_id,
        )
    
    def __del__(self):
        try:
            heartbeat_manager.unregister_processor(self.processor_id)
        except:
            pass
"""]

        # Generate methods
        for method_name, signature in methods:
            # Parse signature to extract parameter names for the JSON payload
            param_names = []
            if signature:
                # Extract parameter names from signature
                for param in signature.split(','):
                    param = param.strip()
                    if param.startswith('*'):
                        continue  # Skip *args and **kwargs for now
                    # Extract just the parameter name (before : or =)
                    param_name = param.split(':')[0].split('=')[0].strip()
                    if param_name and param_name not in ['self']:
                        param_names.append(param_name)
            
            # Build kwargs dict from parameters
            if param_names:
                kwargs_items = ', '.join([f"'{p}': {p}" for p in param_names])
                kwargs_dict = f"{{{kwargs_items}}}"
            else:
                kwargs_dict = "{}"
            
            code_lines.extend([f"""    
    def {method_name}(self{', ' + signature if signature else ''}):
        response = http_post(
            url=f'{{self.server_url}}/call',
            json_data={{
                'processor_id': self.processor_id,
                'function': '{method_name}',
                **{kwargs_dict}
            }}
        )
        response.raise_for_status()
        return response.json()"""
            ])

        return '\n'.join(code_lines)

    # Map base class names to processor types in the server
    processor_type_mapping = {
        'DataLoader': 'dataloader',
        'Dataset': 'dataset',
        'HubOperation': 'hub',
        'Preprocessor': 'preprocessor',
        'InputProcessor': 'processor',
        'Reward': 'reward',
        'Template': 'template',
        'WeightLoader': 'weight_synchronizer',  # Note: server uses 'weight_synchronizer'
    }

    # Scan all modules
    print("Scanning src/twinkle modules...")
    all_classes = {}

    for module_name, module_dir in module_mapping.items():
        module_path = src_twinkle_path / module_dir
        print(f"  Scanning {module_name}...")
        classes = scan_module(module_path)

        for base_class_name, class_list in classes.items():
            if base_class_name not in all_classes:
                all_classes[base_class_name] = {}
            if module_name not in all_classes[base_class_name]:
                all_classes[base_class_name][module_name] = []
            all_classes[base_class_name][module_name].extend(class_list)

    # Generate client files
    print("\nGenerating client classes...")

    # Track all classes per module for __init__.py generation
    module_classes: Dict[str, List[str]] = {}

    for base_class_name, modules in all_classes.items():
        for module_name, class_list in modules.items():
            client_module_path = src_client_path / module_name

            # Create package directory if it doesn't exist
            client_module_path.mkdir(parents=True, exist_ok=True)

            # Track classes for this module
            if module_name not in module_classes:
                module_classes[module_name] = []

            processor_type = processor_type_mapping.get(base_class_name, module_name)

            for class_name, methods in class_list:
                client_file = client_module_path / f'{class_name.lower()}.py'
                print(f"  Generating {client_file}...")

                client_code = generate_client_class(
                    class_name, base_class_name, methods, module_name, processor_type
                )

                # Overwrite the file completely
                with open(client_file, 'w', encoding='utf-8') as f:
                    f.write(client_code)

                # Track class for __init__.py export
                module_classes[module_name].append(class_name)

    # Generate __init__.py files for each module
    print("\nGenerating __init__.py files...")
    for module_name, class_names in module_classes.items():
        client_module_path = src_client_path / module_name
        init_file = client_module_path / '__init__.py'

        # Generate complete __init__.py content
        init_lines = []
        for class_name in sorted(class_names):
            init_lines.append(f"from .{class_name.lower()} import {class_name}")

        init_content = '\n'.join(init_lines) + '\n'

        # Overwrite __init__.py completely
        print(f"  Writing {init_file}...")
        with open(init_file, 'w', encoding='utf-8') as f:
            f.write(init_content)

    print("\nClient generation complete!")
    return all_classes


def generate_models():
    """Generate client wrapper for Model management."""
    from pathlib import Path

    project_root = Path(__file__).parent.parent
    src_client_path = project_root / 'src' / 'client'
    client_module_path = src_client_path / 'model'
    client_module_path.mkdir(parents=True, exist_ok=True)

    model_code = '''from typing import Any, Optional
import uuid
from client.http import TWINKLE_SERVER_URL
from client.http import http_post, heartbeat_manager
import twinkle
from transformers import PretrainedConfig
from twinkle import DeviceMesh


class TransformersModel(twinkle.model.TransformersModel):
    """Client wrapper for TwinkleModel that calls server HTTP endpoints.
    
    This client manages adapters and sends training/inference requests to the model server.
    Each adapter has its own lifecycle managed through automatic heartbeats.
    """
    
    def __init__(self, # noqa
                 model_cls: Optional[Union[Type[PreTrainedModel], str]] = None,
                 pretrained_model_name_or_path: Optional[str] = None,
                 config: Optional[PretrainedConfig] = None,
                 device_mesh: Optional[DeviceMesh] = None,
                 mixed_precision: Literal['no', 'fp8', 'fp16', 'bf16'] = 'bf16',
                 ddp_config: Dict[str, Any] = None,
                 fsdp_config: Dict[str, Any] = None,
                 grad_scaler_config: Dict[str, Any] = None,
                 **kwargs):
        """Initialize model client."""
        self.server_url = TWINKLE_SERVER_URL
        self.adapter_name = None
        kwargs['pretrained_model_name_or_path'] = pretrained_model_name_or_path
        if model_cls:
            if not isinstance(model_cls, str):
                model_cls = model_cls.__name__
            kwargs['model_cls'] = model_cls
        if config is not None:
            kwargs['config'] = config.__dict__
        if device_mesh is not None:
            kwargs['device_mesh'] = device_mesh.__dict__
        kwargs['mixed_precision'] = mixed_precision
        kwargs['ddp_config'] = ddp_config
        kwargs['fsdp_config'] = fsdp_config
        kwargs['grad_scaler_config'] = grad_scaler_config
        response = http_post(
            url=f'{self.server_url}/create',
            json_data=kwargs
        )
        response.raise_for_status()
        return response.json()
    
    def _send_adapter_heartbeat(self):
        """Internal method to send adapter heartbeat."""
        response = http_post(
            url=f'{self.server_url}/heartbeat',
            json_data={'adapter_name': self.adapter_name}
        )
        response.raise_for_status()
    
    def add_adapter(self, adapter_name: str, config: Dict[str, Any]):
        """Add a new adapter to the model and start automatic heartbeat."""
        response = http_post(
            url=f'{self.server_url}/add_adapter',
            request_id=self.request_id,
            authorization=self.authorization,
            json_data={'adapter_name': adapter_name, 'config': config}
        )
        response.raise_for_status()
        
        # Register adapter for automatic heartbeat after successful creation
        self.adapter_name = adapter_name
        heartbeat_manager.register_adapter(
            self.adapter_name,
            self._send_adapter_heartbeat
        )
        
        return response.json()
    
    def __del__(self):
        """Cleanup: unregister adapter from heartbeat manager."""
        try:
            heartbeat_manager.unregister_adapter(self.adapter_name)
        except:
            pass
    
    def forward(self, inputs: Any, **kwargs):
        """Execute forward pass on the model."""
        response = http_post(
            url=f'{self.server_url}/forward',
            json_data={'inputs': inputs, 'adapter_name': self.adapter_name, **kwargs}
        )
        response.raise_for_status()
        return response.json()
    
    def forward_only(self, inputs: Any, **kwargs):
        """Execute forward pass without gradient computation."""
        response = http_post(
            url=f'{self.server_url}/forward_only',
            json_data={'inputs': inputs, 'adapter_name': self.adapter_name, **kwargs}
        )
        response.raise_for_status()
        return response.json()
    
    def calculate_loss(self, **kwargs):
        """Calculate loss from model outputs."""
        response = http_post(
            url=f'{self.server_url}/calculate_loss',
            json_data={'adapter_name': self.adapter_name, **kwargs}
        )
        response.raise_for_status()
        return response.json()
    
    def backward(self, **kwargs):
        """Execute backward pass."""
        response = http_post(
            url=f'{self.server_url}/backward',
            json_data={'adapter_name': self.adapter_name, **kwargs}
        )
        response.raise_for_status()
        return response.json()
    
    def forward_backward(self, inputs: Any, **kwargs):
        """Execute combined forward and backward pass."""
        response = http_post(
            url=f'{self.server_url}/forward_backward',
            json_data={'inputs': inputs, 'adapter_name': self.adapter_name, **kwargs}
        )
        response.raise_for_status()
        return response.json()
    
    def step(self, **kwargs):
        """Execute optimizer step."""
        response = http_post(
            url=f'{self.server_url}/step',
            json_data={'adapter_name': self.adapter_name, **kwargs}
        )
        response.raise_for_status()
        return response.json()
    
    def zero_grad(self, **kwargs):
        """Zero out gradients."""
        response = http_post(
            url=f'{self.server_url}/zero_grad',
            json_data={'adapter_name': self.adapter_name, **kwargs}
        )
        response.raise_for_status()
        return response.json()
    
    def lr_step(self, **kwargs):
        """Execute learning rate scheduler step."""
        response = http_post(
            url=f'{self.server_url}/lr_step',
            json_data={'adapter_name': self.adapter_name, **kwargs}
        )
        response.raise_for_status()
        return response.json()
    
    def set_loss(self, loss_cls: str, **kwargs):
        """Set the loss function."""
        response = http_post(
            url=f'{self.server_url}/set_loss',
            json_data={'loss_cls': loss_cls, 'adapter_name': self.adapter_name, **kwargs}
        )
        response.raise_for_status()
        return response.json()
    
    def set_optimizer(self, optimizer_cls: str, **kwargs):
        """Set the optimizer."""
        response = http_post(
            url=f'{self.server_url}/set_optimizer',
            json_data={'optimizer_cls': optimizer_cls, 'adapter_name': self.adapter_name, **kwargs}
        )
        response.raise_for_status()
        return response.json()
    
    def set_lr_scheduler(self, scheduler_cls: str, **kwargs):
        """Set the learning rate scheduler."""
        response = http_post(
            url=f'{self.server_url}/set_lr_scheduler',
            json_data={'scheduler_cls': scheduler_cls, 'adapter_name': self.adapter_name, **kwargs}
        )
        response.raise_for_status()
        return response.json()
    
    def save(self, output_dir: str, **kwargs):
        """Save model checkpoint."""
        response = http_post(
            url=f'{self.server_url}/save',
            json_data={'output_dir': output_dir, 'adapter_name': self.adapter_name, **kwargs}
        )
        response.raise_for_status()
        return response.json()
    
    def set_template(self, template_cls: str, **kwargs):
        """Set the template for data processing."""
        response = http_post(
            url=f'{self.server_url}/set_template',
            json_data={'template_cls': template_cls, 'adapter_name': self.adapter_name, **kwargs}
        )
        response.raise_for_status()
        return response.json()
    
    def set_processor(self, processor_cls: str, **kwargs):
        """Set the input processor."""
        response = http_post(
            url=f'{self.server_url}/set_processor',
            json_data={'processor_cls': processor_cls, 'adapter_name': self.adapter_name, **kwargs}
        )
        response.raise_for_status()
        return response.json()
'''

    # Create package directory
    client_module_path.mkdir(parents=True, exist_ok=True)

    # Write the model client file (overwrite if exists)
    client_file = client_module_path / 'transformers.py'
    print(f"Generating {client_file}...")
    with open(client_file, 'w', encoding='utf-8') as f:
        f.write(model_code)

    # Create/overwrite __init__.py with exports
    init_file = client_module_path / '__init__.py'
    init_content = "from .transformers import TransformersModel\n"
    print(f"Writing {init_file}...")
    with open(init_file, 'w', encoding='utf-8') as f:
        f.write(init_content)

    print("Model client generation complete!")


def generate_samplers():
    """Generate client wrapper for Sampler management."""
    from pathlib import Path

    project_root = Path(__file__).parent.parent
    src_client_path = project_root / 'src' / 'client'
    client_module_path = src_client_path / 'sampler'
    client_module_path.mkdir(parents=True, exist_ok=True)

    sampler_code = '''from typing import Any, Optional
import uuid
from client.http import TWINKLE_SERVER_URL
from client.http import http_post, heartbeat_manager
import twinkle
import json
from twinkle import DeviceMesh
from peft import PeftConfig
from twinkle.trajectory import Trajectory


class VLLMSampler(twinkle.sampler.VLLMSampler):
    """Client wrapper for Sampler that calls server HTTP endpoints.
    
    This client manages sampling operations and adapter synchronization with the sampler server.
    Each adapter has its own lifecycle managed through automatic heartbeats.
    """
    
    def __init__(self, model_id: str, engine_args: Dict[str, Any], device_mesh: DeviceMesh=None, **kwargs):
        """Create the sampler instance on server."""
        self.server_url = TWINKLE_SERVER_URL
        self.adapter_name = None
        kwargs['model_id'] = model_id
        kwargs['engine_args'] = engine_args
        if device_mesh is not None:
            kwargs['device_mesh'] = device_mesh.__dict__
        response = http_post(
            url=f'{self.server_url}/create',
            json_data=kwargs
        )
        response.raise_for_status()
        return response.json()
    
    def _send_adapter_heartbeat(self):
        """Internal method to send adapter heartbeat."""
        if not self.adapter_name:
            return
        response = http_post(
            url=f'{self.server_url}/heartbeat',
            json_data={'adapter_name': self.adapter_name}
        )
        response.raise_for_status()
    
    def add_adapter_to_sampler(self, adapter_name: str, config: PeftConfig):
        """Add a new adapter to the sampler and start automatic heartbeat.
        
        Args:
            adapter_name: Name of the adapter
            config: LoRA configuration dictionary
        """
        if isinstance(config, PeftConfig):
            config = config.__dict__
        response = http_post(
            url=f'{self.server_url}/add_adapter_to_sampler',
            json_data={'adapter_name': adapter_name, 'config': config}
        )
        response.raise_for_status()
        
        # Register adapter for automatic heartbeat after successful creation
        self.adapter_name = adapter_name
        heartbeat_manager.register_adapter(
            self.adapter_name,
            self._send_adapter_heartbeat
        )
        
        return response.json()
    
    def __del__(self):
        """Cleanup: unregister adapter from heartbeat manager."""
        try:
            if self.adapter_name:
                heartbeat_manager.unregister_adapter(self.adapter_name)
        except:
            pass
    
    def sample(self, trajectories: List[Trajectory], adapter_name: str = '') -> List[Trajectory]:
        """Sample from the model using provided trajectories.
        
        Args:
            trajectories: List of Trajectory objects to sample from
            adapter_name: Optional adapter name (uses instance default if not provided)
        
        Returns:
            List of sampled Trajectory objects
        """
        response = http_post(
            url=f'{self.server_url}/sample',
            json_data={'trajectories': json.dumps(trajectories, ensure_ascii=False), 'adapter_name': adapter_name}
        )
        response.raise_for_status()
        return response.json()
    
    def sync_weights(self, state_dict: Dict[str, Any], adapter_name: str = ''):
        """Synchronize weights to the sampler.
        
        Args:
            state_dict: Model state dictionary to sync
            adapter_name: Optional adapter name (uses instance default if not provided)
        """
        adapter = adapter_name or self.adapter_name
        response = http_post(
            url=f'{self.server_url}/sync_weights',
            json_data={'state_dict': state_dict, 'adapter_name': adapter}
        )
        response.raise_for_status()
        return response.json()
'''

    # Create package directory
    client_module_path.mkdir(parents=True, exist_ok=True)

    # Write the sampler client file (overwrite if exists)
    client_file = client_module_path / 'vllm_sampler.py'
    print(f"Generating {client_file}...")
    with open(client_file, 'w', encoding='utf-8') as f:
        f.write(sampler_code)

    # Create/overwrite __init__.py with exports
    init_file = client_module_path / '__init__.py'
    init_content = "from .vllm_sampler import VLLMSampler\n"
    print(f"Writing {init_file}...")
    with open(init_file, 'w', encoding='utf-8') as f:
        f.write(init_content)
    
    print("Sampler client generation complete!")


if __name__ == '__main__':
    print("Starting client code generation...\n")
    print("=" * 60)
    
    # Generate processor-based clients
    print("\n[1/3] Generating processor-based clients...")
    generate_processors()
    
    # Generate model client
    print("\n" + "=" * 60)
    print("\n[2/3] Generating model client...")
    generate_models()
    
    # Generate sampler client
    print("\n" + "=" * 60)
    print("\n[3/3] Generating sampler client...")
    generate_samplers()
    
    print("\n" + "=" * 60)
    print("\n✓ All client code generation complete!\n")

