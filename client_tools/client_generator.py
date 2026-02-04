# Copyright (c) ModelScope Contributors. All rights reserved.
import ast
from pathlib import Path
from typing import Dict, List, Tuple, Set


def generate_processors():
    """Generate client wrappers for all classes with @remote_function methods."""
    
    # Module mapping: module_name -> directory in src/twinkle
    module_mapping = {
        'dataloader': 'dataloader',
        'dataset': 'dataset',
        'processor': 'processor',
        'reward': 'reward',
        'template': 'template',
        'weight_loader': 'weight_loader',
    }
    
    # Map module names to processor types in the server
    processor_type_mapping = {
        'dataloader': 'dataloader',
        'dataset': 'dataset',
        'processor': 'processor',
        'reward': 'reward',
        'template': 'template',
        'weight_loader': 'weight_loader',
    }
    
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    src_twinkle_path = project_root / 'src' / 'twinkle'
    src_client_path = project_root / 'src' / 'twinkle_client'
    
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

    def extract_typing_imports(signatures: List[str]) -> Set[str]:
        """Extract required typing imports from signatures."""
        typing_patterns = {
            'Union[': 'Union',
            'Optional[': 'Optional',
            'List[': 'List',
            'Dict[': 'Dict',
            'Tuple[': 'Tuple',
            'Type[': 'Type',
            'Any': 'Any',
            'Callable': 'Callable',
            'Literal[': 'Literal',
            'Required[': 'Required',
            'Set[': 'Set',
            'TypedDict': 'TypedDict',
        }

        all_text = ' '.join(signatures)
        return {name for pattern, name in typing_patterns.items() if pattern in all_text}

    def extract_twinkle_imports(signatures: List[str]) -> Set[str]:
        """Extract required twinkle imports from signatures."""
        twinkle_patterns = {
            'InputFeature': ['from twinkle.data_format import InputFeature'],
            'Trajectory': ['from twinkle.data_format import Trajectory'],
            'DataFilter': ['from twinkle.preprocessor import DataFilter'],
            'Preprocessor': ['from twinkle.preprocessor import Preprocessor'],
            'DatasetMeta': ['from twinkle.dataset import DatasetMeta'],
            'Dataset': ['from twinkle.dataset import Dataset'],
            'DeviceMesh': ['from twinkle import DeviceMesh'],
            'Template': ['from twinkle.template import Template'],
            'template.Template': ['from twinkle.template import Template', 'from twinkle import template'],
            'processor.InputProcessor': ['from twinkle.processor import InputProcessor', 'from twinkle import processor'],
            'InputProcessor': ['from twinkle.processor import InputProcessor'],
        }

        all_text = ' '.join(signatures)
        imports = set()
        for pattern, stmts in twinkle_patterns.items():
            if pattern in all_text:
                imports.update(stmts)

        return imports

    def parse_params_from_signature(signature: str) -> List[str]:
        """Parse parameter names from signature, handling nested brackets."""
        params = []
        current = ''
        depth = 0

        for char in signature + ',':
            if char in '[(':
                depth += 1
            elif char in '])':
                depth -= 1

            if char == ',' and depth == 0:
                name = current.split(':')[0].split('=')[0].strip()
                if name and name != 'self' and not name.startswith('*'):
                    params.append(name)
                current = ''
            else:
                current += char

        return params

    def find_classes_with_remote_methods(file_path: Path) -> List[Tuple[str, str, List[Tuple[str, str]]]]:
        """Find all classes that have @remote_function decorated methods."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read(), filename=str(file_path))
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return []

        def has_remote_decorator(func: ast.FunctionDef) -> bool:
            for dec in func.decorator_list:
                if isinstance(dec, ast.Name) and dec.id == 'remote_function':
                    return True
                if isinstance(dec, ast.Call):
                    func_node = dec.func
                    if isinstance(func_node, ast.Name) and func_node.id == 'remote_function':
                        return True
                    if isinstance(func_node, ast.Attribute) and func_node.attr == 'remote_function':
                        return True
            return False

        def is_public_or_dunder(name: str) -> bool:
            return (name.startswith('__') and name.endswith('__')) or not name.startswith('_')

        def get_base_name(node: ast.ClassDef) -> str:
            if not node.bases:
                return 'object'
            base = node.bases[0]
            if isinstance(base, ast.Name):
                return base.id
            if isinstance(base, ast.Attribute):
                return base.attr
            return 'object'

        classes_found = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue

            methods = [
                (item.name, get_method_signature(item))
                for item in node.body
                if isinstance(item, ast.FunctionDef)
                   and has_remote_decorator(item)
                   and is_public_or_dunder(item.name)
            ]

            # Extract __init__ signature separately (it may not have @remote_function)
            init_signature = ''
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                    init_signature = get_method_signature(item)
                    break

            if methods:
                classes_found.append((node.name, get_base_name(node), methods, init_signature))

        return classes_found

    def generate_client_class(class_name: str, base_class_name: str,
                              methods: List[Tuple[str, str]], module_name: str,
                              processor_type: str, source_filename: str,
                              has_base_file: bool, init_signature: str = '') -> str:
        """Generate client wrapper class code."""

        def build_imports() -> Tuple[List[str], str]:
            # Include both method signatures and __init__ signature for import detection
            signatures = [sig for _, sig in methods]
            if init_signature:
                signatures.append(init_signature)
            
            typing_imports = extract_typing_imports(signatures)
            twinkle_imports = extract_twinkle_imports(signatures)

            lines = []
            if typing_imports:
                lines.append(f"from typing import {', '.join(sorted(typing_imports))}")
            lines.extend([
                "from twinkle_client.http import TWINKLE_SERVER_URL",
                "from twinkle_client.http import http_post, heartbeat_manager",
            ])
            lines.extend(sorted(twinkle_imports))

            if source_filename == 'base':
                inheritance = "object"
            elif base_class_name == 'IterableDataset':
                lines.append("from torch.utils.data import IterableDataset")
                inheritance = "IterableDataset"
            elif has_base_file and base_class_name != 'object':
                lines.append(f"from .base import {base_class_name}")
                inheritance = base_class_name
            else:
                inheritance = "object"

            lines.append("")
            return lines, inheritance

        def build_method(name: str, signature: str) -> str:
            param_names = parse_params_from_signature(signature)
            kwargs_dict = '{' + ', '.join(f"'{p}': {p}" for p in param_names) + '}' if param_names else '{}'
            sig_part = f', {signature}' if signature else ''
            if 'kwargs' in sig_part:
                extra_args = '\n                **kwargs'
            else:
                extra_args = ''
            ret = 'self' if name == '__iter__' else 'response.json()["result"]'

            code = f'''    
    def {name}(self{sig_part}):
        response = http_post(
            url=f'{{self.server_url}}/processors/call',
            json_data={{
                'processor_id': self.processor_id,
                'function': '{name}',
                **{kwargs_dict},{extra_args}
            }}
        )
        response.raise_for_status()
        return {ret}
    '''
            if name == '__iter__':
                code += '''
    def __next__(self):
        response = http_post(
            url=f'{self.server_url}/processors/call',
            json_data={
                'processor_id': self.processor_id,
                'function': '__next__',
            }
        )
        response.raise_for_status()
        return response.json()["result"]
    '''
            return code

        import_lines, inheritance = build_imports()

        # Build __init__ method with actual signature
        if init_signature:
            # Extract parameter names from signature (excluding **kwargs)
            param_names = parse_params_from_signature(init_signature)
            init_params = f"self, {init_signature}" if init_signature else "self"
            
            # Check if signature has **kwargs
            has_kwargs = '**' in init_signature
            
            # Extract the **kwargs name if present
            kwargs_name = None
            if has_kwargs:
                # Find the **kwargs parameter name
                for part in init_signature.split(','):
                    part = part.strip()
                    if part.startswith('**'):
                        # Extract name after **, before : or end
                        kwargs_name = part[2:].split(':')[0].strip()
                        break
            
            # Build kwargs dict for HTTP request
            if param_names:
                kwargs_items = ', '.join([f"'{p}': {p}" for p in param_names])
                if has_kwargs and kwargs_name:
                    # Include both named params and **kwargs
                    kwargs_dict = f"{{{kwargs_items}}}, **{kwargs_name}"
                else:
                    kwargs_dict = f"{{{kwargs_items}}}"
            else:
                if has_kwargs and kwargs_name:
                    kwargs_dict = kwargs_name
                else:
                    kwargs_dict = "{}"
        else:
            # Fallback to **kwargs if no __init__ found
            init_params = "self, **kwargs"
            kwargs_dict = "kwargs"

        class_template = f'''{chr(10).join(import_lines)}
class {class_name}({inheritance}):
    """Client wrapper for {class_name} that calls server HTTP endpoints."""

    def __init__({init_params}):
        from twinkle_client.http import get_base_url
        self.server_url = get_base_url()

        response = http_post(
            url=f'{{self.server_url}}/processors/create',
            json_data={{
                'processor_type': '{processor_type}',
                'class_type': '{class_name}',
                **{kwargs_dict}
            }}
        )
        response.raise_for_status()
        self.processor_id = response.json()['processor_id']
        heartbeat_manager.register_processor(self.processor_id)

    def __del__(self):
        try:
            heartbeat_manager.unregister_processor(self.processor_id)
        except:
            pass

    '''

        method_codes = [build_method(name, sig) for name, sig in methods]

        return class_template + '\n'.join(method_codes)

    def scan_modules(src_twinkle_path: Path, module_mapping: Dict[str, str]) -> Dict:
        """Scan all modules for classes with @remote_function methods."""
        print("Scanning src/twinkle modules for classes with @remote_function methods...")

        module_files = {}
        for module_name, module_dir in module_mapping.items():
            module_path = src_twinkle_path / module_dir
            if not module_path.exists():
                continue

            print(f"  Scanning {module_name}...")
            for py_file in module_path.glob('*.py'):
                if py_file.name.startswith('_'):
                    continue

                if classes := find_classes_with_remote_methods(py_file):
                    module_files.setdefault(module_name, {}).setdefault(py_file.stem, []).extend(classes)

        return module_files

    def write_client_files(module_files: Dict, src_client_path: Path,
                           processor_type_mapping: Dict[str, str]) -> None:
        """Generate and write client files."""
        print("\nGenerating client classes...")

        for module_name, source_files in module_files.items():
            client_module_path = src_client_path / module_name
            client_module_path.mkdir(parents=True, exist_ok=True)

            processor_type = processor_type_mapping.get(module_name, module_name)
            has_base_file = 'base' in source_files

            for source_filename, classes in source_files.items():
                client_file = client_module_path / f'{source_filename}.py'
                print(f"  Writing {client_file}...")

                code = '\n\n'.join(
                    generate_client_class(class_name, base_class_name, methods,
                                          module_name, processor_type, source_filename, has_base_file, init_signature)
                    for class_name, base_class_name, methods, init_signature in classes
                )
                client_file.write_text(code, encoding='utf-8')

    def write_init_files(module_files: Dict, src_client_path: Path) -> None:
        """Generate __init__.py files for each module."""
        print("\nGenerating __init__.py files...")

        for module_name, source_files in module_files.items():
            init_file = src_client_path / module_name / '__init__.py'
            print(f"  Writing {init_file}...")

            init_lines = [
                f"from .{source_filename} import {class_name}"
                for source_filename, classes in sorted(source_files.items())
                for class_name, _, _, _ in classes
            ]
            init_file.write_text('\n'.join(sorted(init_lines)) + '\n', encoding='utf-8')

    module_files = scan_modules(src_twinkle_path, module_mapping)
    write_client_files(module_files, src_client_path, processor_type_mapping)
    write_init_files(module_files, src_client_path)
    print("\nProcessor client generation complete!")
    return module_files


def generate_models():
    """Generate client wrapper for Model management."""
    from pathlib import Path

    project_root = Path(__file__).parent.parent
    src_client_path = project_root / 'src' / 'twinkle_client'
    client_module_path = src_client_path / 'model'
    client_module_path.mkdir(parents=True, exist_ok=True)

    model_code = '''from typing import Any, Optional, Union, Type, Dict, Literal, List
import uuid
from twinkle_client.http import TWINKLE_SERVER_URL
from twinkle_client.http import http_post, heartbeat_manager
from twinkle import DeviceMesh
from twinkle.data_format import InputFeature, Trajectory


class MultiLoraTransformersModel:
    """Client wrapper for TwinkleModel that calls server HTTP endpoints.
    
    This client manages adapters and sends training/inference requests to the model server.
    Each adapter has its own lifecycle managed through automatic heartbeats.
    """
    
    def __init__(self, model_id: str, **kwargs):
        """Initialize model client."""
        from twinkle_client.http import get_base_url
        self.server_url = get_base_url()
        
        self.model_id = model_id
        if '://' in model_id:
            model_id = model_id.split('://')[1]
        self.server_url = f'{self.server_url}/models/{model_id}'
        self.adapter_name = None
        response = http_post(
            url=f'{self.server_url}/create',
        )
        response.raise_for_status()
    
    def _send_adapter_heartbeat(self):
        """Internal method to send adapter heartbeat."""
        response = http_post(
            url=f'{self.server_url}/heartbeat',
            json_data={'adapter_name': self.adapter_name}
        )
        response.raise_for_status()
    
    def add_adapter_to_model(self, adapter_name: str, config: Dict[str, Any], **kwargs):
        """Add a new adapter to the model and start automatic heartbeat."""
        response = http_post(
            url=f'{self.server_url}/add_adapter_to_model',
            json_data={'adapter_name': adapter_name, 'config': config, **kwargs}
        )
        response.raise_for_status()
        
        # Register adapter for automatic heartbeat after successful creation
        self.adapter_name = adapter_name
        heartbeat_manager.register_adapter(
            self.adapter_name,
            self._send_adapter_heartbeat
        )
    
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
        return response.json()['result']
    
    def forward_only(self, inputs: Any, **kwargs):
        """Execute forward pass without gradient computation."""
        response = http_post(
            url=f'{self.server_url}/forward_only',
            json_data={'inputs': inputs, 'adapter_name': self.adapter_name, **kwargs}
        )
        response.raise_for_status()
        return response.json()['result']
    
    def calculate_loss(self, **kwargs):
        """Calculate loss from model outputs."""
        response = http_post(
            url=f'{self.server_url}/calculate_loss',
            json_data={'adapter_name': self.adapter_name, **kwargs}
        )
        response.raise_for_status()
        return response.json()['result']
    
    def get_train_configs(self, **kwargs):
        """Get training configs"""
        response = http_post(
            url=f'{self.server_url}/get_train_configs',
            json_data={'adapter_name': self.adapter_name, **kwargs}
        )
        response.raise_for_status()
        return response.json()['result']

    def backward(self, **kwargs):
        """Execute backward pass."""
        response = http_post(
            url=f'{self.server_url}/backward',
            json_data={'adapter_name': self.adapter_name, **kwargs}
        )
        response.raise_for_status()
        return response.json()['result']
    
    def forward_backward(self, inputs: Any, **kwargs):
        """Execute combined forward and backward pass."""
        response = http_post(
            url=f'{self.server_url}/forward_backward',
            json_data={'inputs': inputs, 'adapter_name': self.adapter_name, **kwargs}
        )
        response.raise_for_status()
        return response.json()['result']
    
    def step(self, **kwargs):
        """Execute optimizer step."""
        response = http_post(
            url=f'{self.server_url}/step',
            json_data={'adapter_name': self.adapter_name, **kwargs}
        )
        response.raise_for_status()
        return response.json()['result']
    
    def zero_grad(self, **kwargs):
        """Zero out gradients."""
        response = http_post(
            url=f'{self.server_url}/zero_grad',
            json_data={'adapter_name': self.adapter_name, **kwargs}
        )
        response.raise_for_status()
        return response.json()['result']
    
    def lr_step(self, **kwargs):
        """Execute learning rate scheduler step."""
        response = http_post(
            url=f'{self.server_url}/lr_step',
            json_data={'adapter_name': self.adapter_name, **kwargs}
        )
        response.raise_for_status()
        return response.json()['result']
    
    def set_loss(self, loss_cls: str, **kwargs):
        """Set the loss function."""
        response = http_post(
            url=f'{self.server_url}/set_loss',
            json_data={'loss_cls': loss_cls, 'adapter_name': self.adapter_name, **kwargs}
        )
        response.raise_for_status()
        return response.json()['result']

    def clip_grad_norm(self, max_grad_norm: float=1.0, norm_type=2, **kwargs):
        """Set the loss function."""
        response = http_post(
            url=f'{self.server_url}/clip_grad_norm',
            json_data={'max_grad_norm': max_grad_norm, 'norm_type': norm_type, 'adapter_name': self.adapter_name, **kwargs}
        )
        response.raise_for_status()
        return response.json()['result']

    def set_optimizer(self, optimizer_cls: str, **kwargs):
        """Set the optimizer."""
        response = http_post(
            url=f'{self.server_url}/set_optimizer',
            json_data={'optimizer_cls': optimizer_cls, 'adapter_name': self.adapter_name, **kwargs}
        )
        response.raise_for_status()
        return response.json()['result']
    
    def set_lr_scheduler(self, scheduler_cls: str, **kwargs):
        """Set the learning rate scheduler."""
        response = http_post(
            url=f'{self.server_url}/set_lr_scheduler',
            json_data={'scheduler_cls': scheduler_cls, 'adapter_name': self.adapter_name, **kwargs}
        )
        response.raise_for_status()
        return response.json()['result']
    
    def save(self, name: str, **kwargs):
        """Save model checkpoint."""
        response = http_post(
            url=f'{self.server_url}/save',
            json_data={'name': name, 'adapter_name': self.adapter_name, **kwargs}
        )
        response.raise_for_status()
        return response.json()['result']
        
    def load(self, name: str, **kwargs):
        """Load model checkpoint."""
        response = http_post(
            url=f'{self.server_url}/load',
            json_data={'name': name, 'adapter_name': self.adapter_name, **kwargs}
        )
        response.raise_for_status()
        return response.json()['result']
    
    def set_template(self, template_cls: str, **kwargs):
        """Set the template for data processing."""
        response = http_post(
            url=f'{self.server_url}/set_template',
            json_data={'template_cls': template_cls, 'adapter_name': self.adapter_name, 'model_id': self.model_id, **kwargs}
        )
        response.raise_for_status()
        return response.json()['result']
    
    def set_processor(self, processor_cls: str, **kwargs):
        """Set the input processor."""
        response = http_post(
            url=f'{self.server_url}/set_processor',
            json_data={'processor_cls': processor_cls, 'adapter_name': self.adapter_name, **kwargs}
        )
        response.raise_for_status()
        return response.json()['result']

    def calculate_metric(self, is_training: bool = True, **kwargs):
        """Calculate metrics from model outputs."""
        response = http_post(
            url=f'{self.server_url}/calculate_metric',
            json_data={'is_training': is_training, 'adapter_name': self.adapter_name, **kwargs}
        )
        response.raise_for_status()
        return response.json()['result']

    def get_state_dict(self, **kwargs):
        """Get model state dictionary."""
        response = http_post(
            url=f'{self.server_url}/get_state_dict',
            json_data={'adapter_name': self.adapter_name, **kwargs}
        )
        response.raise_for_status()
        return response.json()['result']

    def upload_to_hub(self, checkpoint_dir: str, hub_model_id: str, hub_token: Optional[str] = None, async_upload: bool = True):
        """Upload model checkpoint to hub.
        
        Args:
            checkpoint_dir: The directory path of the checkpoint to upload.
            hub_model_id: The hub model id.
            hub_token: The hub token (optional).
            async_upload: Whether to use async upload (default: True).
        """
        response = http_post(
            url=f'{self.server_url}/upload_to_hub',
            json_data={
                'checkpoint_dir': checkpoint_dir,
                'hub_model_id': hub_model_id,
                'hub_token': hub_token,
                'async_upload': async_upload
            }
        )
        response.raise_for_status()
        return response.json()
'''

    # Write the model client file
    client_file = client_module_path / 'multi_lora_transformers.py'
    print(f"Generating {client_file}...")
    with open(client_file, 'w', encoding='utf-8') as f:
        f.write(model_code)

    # Create/overwrite __init__.py
    init_file = client_module_path / '__init__.py'
    init_content = "from .multi_lora_transformers import MultiLoraTransformersModel\n"
    print(f"Writing {init_file}...")
    with open(init_file, 'w', encoding='utf-8') as f:
        f.write(init_content)

    print("Model client generation complete!")


def generate_samplers():
    """Generate client wrapper for Sampler management."""
    from pathlib import Path

    project_root = Path(__file__).parent.parent
    src_client_path = project_root / 'src' / 'twinkle_client'
    client_module_path = src_client_path / 'sampler'
    client_module_path.mkdir(parents=True, exist_ok=True)

    sampler_code = '''from typing import Any, Optional, List, Dict, Union
import uuid
from twinkle_client.http import TWINKLE_SERVER_URL
from twinkle_client.http import http_post, heartbeat_manager
from twinkle.sampler.base import Sampler
from twinkle.sampler.types import SamplingParams, SampleResponse
from twinkle import DeviceMesh
from peft import PeftConfig
from twinkle.data_format import Trajectory, InputFeature
import json


class VLLMSampler(Sampler):
    """Client wrapper for Sampler that calls server HTTP endpoints.
    
    This client manages sampling operations and adapter synchronization with the sampler server.
    Each adapter has its own lifecycle managed through automatic heartbeats.
    """
    
    def __init__(self, model_id: str, **kwargs):
        """Create the sampler instance on server."""
        from twinkle_client.http import get_base_url
        self.server_url = get_base_url()
        
        self.adapter_name = None
        if '://' in model_id:
            model_id = model_id.split('://')[1]
        self.server_url = f'{self.server_url}/models/{model_id}'
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
    
    def add_adapter_to_sampler(self, adapter_name: str, config: PeftConfig, **kwargs):
        """Add a new adapter to the sampler and start automatic heartbeat."""
        if isinstance(config, PeftConfig):
            config = config.__dict__
        response = http_post(
            url=f'{self.server_url}/add_adapter_to_sampler',
            json_data={'adapter_name': adapter_name, 'config': config, **kwargs}
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
    
    def sample(
        self,
        inputs: Union[List[Trajectory], List[InputFeature]],
        sampling_params: Optional[Dict[str, Any]] = None,
        adapter_name: str = ''
    ) -> SampleResponse:
        """Sample from the model.
        
        Args:
            inputs: List of Trajectory or InputFeature to sample from.
            sampling_params: Sampling parameters dict.
            adapter_name: Adapter name.
            
        Returns:
            SampleResponse with sampled sequences.
        """
        response = http_post(
            url=f'{self.server_url}/sample',
            json_data={
                'inputs': inputs,
                'sampling_params': sampling_params,
                'adapter_name': adapter_name
            }
        )
        response.raise_for_status()
        return response.json()
    
    def sync_weights(self, state_dict: Dict[str, Any], adapter_name: str = ''):
        """Synchronize weights to the sampler."""
        adapter = adapter_name or self.adapter_name
        response = http_post(
            url=f'{self.server_url}/sync_weights',
            json_data={'state_dict': state_dict, 'adapter_name': adapter}
        )
        response.raise_for_status()
        return response.json()
    
    def set_template(self, template_cls: str, adapter_name: str = '', **kwargs):
        """Set the template for encoding trajectories."""
        response = http_post(
            url=f'{self.server_url}/set_template',
            json_data={'template_cls': template_cls, 'adapter_name': adapter_name, **kwargs}
        )
        response.raise_for_status()
        return response.json()
'''

    # Write the sampler client file
    client_file = client_module_path / 'vllm_sampler.py'
    print(f"Generating {client_file}...")
    with open(client_file, 'w', encoding='utf-8') as f:
        f.write(sampler_code)

    # Create/overwrite __init__.py
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
