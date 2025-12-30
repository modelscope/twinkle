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
    
    def find_classes_in_file(file_path: Path) -> List[Tuple[str, str, List[str]]]:
        """Find classes that inherit from base classes.
        
        Returns:
            List of tuples (class_name, base_class_name, method_names)
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read(), filename=str(file_path))
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return []
        
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
                                    methods.append(item.name)
                        
                        classes_found.append((node.name, base_name, methods))
        
        return classes_found
    
    def scan_module(module_name: str, module_path: Path) -> Dict[str, List[Tuple[str, List[str]]]]:
        """Scan a module directory for classes.
        
        Returns:
            Dict mapping base_class_name to list of (class_name, methods)
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
    
    def generate_client_class(class_name: str, base_class_name: str, methods: List[str], 
                             module_name: str, processor_type: str) -> str:
        """Generate client wrapper class code."""
        code_lines = [
            f"from typing import Any, Optional",
            f"import uuid",
            f"from client.http.http_utils import http_post\n",
            f"\n",
            f"class {class_name}:",
            f"    \"\"\"Client wrapper for {class_name} that calls server HTTP endpoints.\"\"\"",
            f"    ",
            f"    def __init__(self, server_url: str, request_id: Optional[str] = None, ",
            f"                 authorization: Optional[str] = None, **kwargs):",
            f"        self.server_url = server_url",
            f"        self.request_id = request_id or str(uuid.uuid4().hex)",
            f"        self.authorization = authorization or 'Bearer default_token'",
            f"        ",
            f"        # Create processor instance on server",
            f"        response = http_post(",
            f"            url=f'{{self.server_url}}/create',",
            f"            request_id=self.request_id,",
            f"            authorization=self.authorization,",
            f"            json_data={{",
            f"                'processor_type': '{processor_type}',",
            f"                'class_type': '{class_name}',",
            f"                **kwargs",
            f"            }}",
            f"        )",
            f"        response.raise_for_status()",
            f"        self.processor_id = response.json()",
            f"    ",
            f"    def heartbeat(self):",
            f"        \"\"\"Send heartbeat to keep the processor alive.\"\"\"",
            f"        response = http_post(",
            f"            url=f'{{self.server_url}}/heartbeat',",
            f"            request_id=self.request_id,",
            f"            authorization=self.authorization,",
            f"            json_data={{'processor_id': self.processor_id}}",
            f"        )",
            f"        response.raise_for_status()",
            f"        return response.json()",
        ]
        
        # Generate methods
        for method_name in methods:
            code_lines.extend([
                f"    ",
                f"    def {method_name}(self, **kwargs):",
                f"        \"\"\"Call {method_name} on the remote processor.\"\"\"",
                f"        response = http_post(",
                f"            url=f'{{self.server_url}}/call',",
                f"            request_id=self.request_id,",
                f"            authorization=self.authorization,",
                f"            json_data={{",
                f"                'processor_id': self.processor_id,",
                f"                'function': '{method_name}',",
                f"                **kwargs",
                f"            }}",
                f"        )",
                f"        response.raise_for_status()",
                f"        return response.json()",
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
        classes = scan_module(module_name, module_path)
        
        for base_class_name, class_list in classes.items():
            if base_class_name not in all_classes:
                all_classes[base_class_name] = {}
            if module_name not in all_classes[base_class_name]:
                all_classes[base_class_name][module_name] = []
            all_classes[base_class_name][module_name].extend(class_list)
    
    # Generate client files
    print("\nGenerating client classes...")
    
    for base_class_name, modules in all_classes.items():
        for module_name, class_list in modules.items():
            client_module_path = src_client_path / module_name
            client_module_path.mkdir(parents=True, exist_ok=True)
            
            # Create __init__.py if it doesn't exist
            init_file = client_module_path / '__init__.py'
            if not init_file.exists():
                init_file.write_text('')
            
            processor_type = processor_type_mapping.get(base_class_name, module_name)
            
            for class_name, methods in class_list:
                client_file = client_module_path / f'{class_name.lower()}.py'
                print(f"  Generating {client_file}...")
                
                client_code = generate_client_class(
                    class_name, base_class_name, methods, module_name, processor_type
                )
                
                with open(client_file, 'w', encoding='utf-8') as f:
                    f.write(client_code)
                
                # Update __init__.py to export the class
                init_content = init_file.read_text()
                import_line = f"from .{class_name.lower()} import {class_name}\n"
                if import_line not in init_content:
                    with open(init_file, 'a', encoding='utf-8') as f:
                        f.write(import_line)
    
    print("\nClient generation complete!")
    return all_classes


def generate_models():
    """Generate client wrapper for Model management."""
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent
    src_client_path = project_root / 'src' / 'client'
    client_module_path = src_client_path / 'model'
    client_module_path.mkdir(parents=True, exist_ok=True)
    
    # Model endpoints from src/adapter/twinkle/model.py
    model_methods = [
        'forward', 'forward_only', 'calculate_loss', 'backward', 
        'forward_backward', 'step', 'zero_grad', 'lr_step',
        'set_loss', 'set_optimizer', 'set_lr_scheduler', 'save',
        'add_adapter', 'set_template', 'set_processor'
    ]
    
    model_code = '''from typing import Any, Dict, List, Union, Optional
import uuid
from client.http.http_utils import http_post


class TwinkleModelClient:
    """Client wrapper for TwinkleModel that calls server HTTP endpoints.
    
    This client manages adapters and sends training/inference requests to the model server.
    Each adapter has its own lifecycle managed through heartbeats.
    """
    
    def __init__(self, server_url: str, adapter_name: str,
                 request_id: Optional[str] = None,
                 authorization: Optional[str] = None):
        """Initialize model client.
        
        Args:
            server_url: Base URL of the model server
            adapter_name: Name of the adapter to use
            request_id: Optional request ID for X-Ray-Serve-Request-Id header
            authorization: Optional authorization token
        """
        self.server_url = server_url
        self.adapter_name = adapter_name
        self.request_id = request_id or str(uuid.uuid4().hex)
        self.authorization = authorization or 'Bearer default_token'
    
    def create(self, **kwargs):
        """Create the model instance on server."""
        response = http_post(
            url=f'{self.server_url}/create',
            request_id=self.request_id,
            authorization=self.authorization,
            json_data=kwargs
        )
        response.raise_for_status()
        return response.json()
    
    def heartbeat(self):
        """Send heartbeat to keep the adapter alive."""
        response = http_post(
            url=f'{self.server_url}/heartbeat',
            request_id=self.request_id,
            authorization=self.authorization,
            json_data={'adapter_name': self.adapter_name}
        )
        response.raise_for_status()
        return response.json()
    
    def forward(self, inputs: Any, **kwargs):
        """Execute forward pass on the model."""
        response = http_post(
            url=f'{self.server_url}/forward',
            request_id=self.request_id,
            authorization=self.authorization,
            json_data={'inputs': inputs, 'adapter_name': self.adapter_name, **kwargs}
        )
        response.raise_for_status()
        return response.json()
    
    def forward_only(self, inputs: Any, **kwargs):
        """Execute forward pass without gradient computation."""
        response = http_post(
            url=f'{self.server_url}/forward_only',
            request_id=self.request_id,
            authorization=self.authorization,
            json_data={'inputs': inputs, 'adapter_name': self.adapter_name, **kwargs}
        )
        response.raise_for_status()
        return response.json()
    
    def calculate_loss(self, **kwargs):
        """Calculate loss from model outputs."""
        response = http_post(
            url=f'{self.server_url}/calculate_loss',
            request_id=self.request_id,
            authorization=self.authorization,
            json_data={'adapter_name': self.adapter_name, **kwargs}
        )
        response.raise_for_status()
        return response.json()
    
    def backward(self, **kwargs):
        """Execute backward pass."""
        response = http_post(
            url=f'{self.server_url}/backward',
            request_id=self.request_id,
            authorization=self.authorization,
            json_data={'adapter_name': self.adapter_name, **kwargs}
        )
        response.raise_for_status()
        return response.json()
    
    def forward_backward(self, inputs: Any, **kwargs):
        """Execute combined forward and backward pass."""
        response = http_post(
            url=f'{self.server_url}/forward_backward',
            request_id=self.request_id,
            authorization=self.authorization,
            json_data={'inputs': inputs, 'adapter_name': self.adapter_name, **kwargs}
        )
        response.raise_for_status()
        return response.json()
    
    def step(self, **kwargs):
        """Execute optimizer step."""
        response = http_post(
            url=f'{self.server_url}/step',
            request_id=self.request_id,
            authorization=self.authorization,
            json_data={'adapter_name': self.adapter_name, **kwargs}
        )
        response.raise_for_status()
        return response.json()
    
    def zero_grad(self, **kwargs):
        """Zero out gradients."""
        response = http_post(
            url=f'{self.server_url}/zero_grad',
            request_id=self.request_id,
            authorization=self.authorization,
            json_data={'adapter_name': self.adapter_name, **kwargs}
        )
        response.raise_for_status()
        return response.json()
    
    def lr_step(self, **kwargs):
        """Execute learning rate scheduler step."""
        response = http_post(
            url=f'{self.server_url}/lr_step',
            request_id=self.request_id,
            authorization=self.authorization,
            json_data={'adapter_name': self.adapter_name, **kwargs}
        )
        response.raise_for_status()
        return response.json()
    
    def set_loss(self, loss_cls: str, **kwargs):
        """Set the loss function."""
        response = http_post(
            url=f'{self.server_url}/set_loss',
            request_id=self.request_id,
            authorization=self.authorization,
            json_data={'loss_cls': loss_cls, 'adapter_name': self.adapter_name, **kwargs}
        )
        response.raise_for_status()
        return response.json()
    
    def set_optimizer(self, optimizer_cls: str, **kwargs):
        """Set the optimizer."""
        response = http_post(
            url=f'{self.server_url}/set_optimizer',
            request_id=self.request_id,
            authorization=self.authorization,
            json_data={'optimizer_cls': optimizer_cls, 'adapter_name': self.adapter_name, **kwargs}
        )
        response.raise_for_status()
        return response.json()
    
    def set_lr_scheduler(self, scheduler_cls: str, **kwargs):
        """Set the learning rate scheduler."""
        response = http_post(
            url=f'{self.server_url}/set_lr_scheduler',
            request_id=self.request_id,
            authorization=self.authorization,
            json_data={'scheduler_cls': scheduler_cls, 'adapter_name': self.adapter_name, **kwargs}
        )
        response.raise_for_status()
        return response.json()
    
    def save(self, output_dir: str, **kwargs):
        """Save model checkpoint."""
        response = http_post(
            url=f'{self.server_url}/save',
            request_id=self.request_id,
            authorization=self.authorization,
            json_data={'output_dir': output_dir, 'adapter_name': self.adapter_name, **kwargs}
        )
        response.raise_for_status()
        return response.json()
    
    def add_adapter(self, adapter_name: str, config: Dict[str, Any]):
        """Add a new adapter to the model."""
        response = http_post(
            url=f'{self.server_url}/add_adapter',
            request_id=self.request_id,
            authorization=self.authorization,
            json_data={'adapter_name': adapter_name, 'config': config}
        )
        response.raise_for_status()
        return response.json()
    
    def set_template(self, template_cls: str, **kwargs):
        """Set the template for data processing."""
        response = http_post(
            url=f'{self.server_url}/set_template',
            request_id=self.request_id,
            authorization=self.authorization,
            json_data={'template_cls': template_cls, 'adapter_name': self.adapter_name, **kwargs}
        )
        response.raise_for_status()
        return response.json()
    
    def set_processor(self, processor_cls: str, **kwargs):
        """Set the input processor."""
        response = http_post(
            url=f'{self.server_url}/set_processor',
            request_id=self.request_id,
            authorization=self.authorization,
            json_data={'processor_cls': processor_cls, 'adapter_name': self.adapter_name, **kwargs}
        )
        response.raise_for_status()
        return response.json()
'''
    
    # Write the model client file
    client_file = client_module_path / 'twinklemodelclient.py'
    print(f"Generating {client_file}...")
    with open(client_file, 'w', encoding='utf-8') as f:
        f.write(model_code)
    
    # Update __init__.py
    init_file = client_module_path / '__init__.py'
    if not init_file.exists():
        init_file.write_text('')
    
    init_content = init_file.read_text()
    import_line = "from .twinklemodelclient import TwinkleModelClient\n"
    if import_line not in init_content:
        with open(init_file, 'a', encoding='utf-8') as f:
            f.write(import_line)
    
    print("Model client generation complete!")


def generate_samplers():
    """Generate client wrapper for Sampler management."""
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent
    src_client_path = project_root / 'src' / 'client'
    client_module_path = src_client_path / 'sampler'
    client_module_path.mkdir(parents=True, exist_ok=True)
    
    sampler_code = '''from typing import Any, Dict, List, Optional
import uuid
from client.http.http_utils import http_post


class SamplerClient:
    """Client wrapper for Sampler that calls server HTTP endpoints.
    
    This client manages sampling operations and adapter synchronization with the sampler server.
    """
    
    def __init__(self, server_url: str, adapter_name: str = '',
                 request_id: Optional[str] = None,
                 authorization: Optional[str] = None):
        """Initialize sampler client.
        
        Args:
            server_url: Base URL of the sampler server
            adapter_name: Name of the adapter to use (optional)
            request_id: Optional request ID for X-Ray-Serve-Request-Id header
            authorization: Optional authorization token
        """
        self.server_url = server_url
        self.adapter_name = adapter_name
        self.request_id = request_id or str(uuid.uuid4().hex)
        self.authorization = authorization or 'Bearer default_token'
    
    def create(self, **kwargs):
        """Create the sampler instance on server."""
        response = http_post(
            url=f'{self.server_url}/create',
            request_id=self.request_id,
            authorization=self.authorization,
            json_data=kwargs
        )
        response.raise_for_status()
        return response.json()
    
    def heartbeat(self):
        """Send heartbeat to keep the adapter alive."""
        if not self.adapter_name:
            raise ValueError("adapter_name must be set for heartbeat")
        response = http_post(
            url=f'{self.server_url}/heartbeat',
            request_id=self.request_id,
            authorization=self.authorization,
            json_data={'adapter_name': self.adapter_name}
        )
        response.raise_for_status()
        return response.json()
    
    def sample(self, trajectories: List[Any], adapter_name: str = '') -> List[Any]:
        """Sample from the model using provided trajectories.
        
        Args:
            trajectories: List of Trajectory objects to sample from
            adapter_name: Optional adapter name (uses instance default if not provided)
        
        Returns:
            List of sampled Trajectory objects
        """
        adapter = adapter_name or self.adapter_name
        response = http_post(
            url=f'{self.server_url}/sample',
            request_id=self.request_id,
            authorization=self.authorization,
            json_data={'trajectories': trajectories, 'adapter_name': adapter}
        )
        response.raise_for_status()
        return response.json()
    
    def add_adapter_to_sampler(self, adapter_name: str, config: Dict[str, Any]):
        """Add a new adapter to the sampler.
        
        Args:
            adapter_name: Name of the adapter
            config: LoRA configuration dictionary
        """
        response = http_post(
            url=f'{self.server_url}/add_adapter_to_sampler',
            request_id=self.request_id,
            authorization=self.authorization,
            json_data={'adapter_name': adapter_name, 'config': config}
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
            request_id=self.request_id,
            authorization=self.authorization,
            json_data={'state_dict': state_dict, 'adapter_name': adapter}
        )
        response.raise_for_status()
        return response.json()
'''
    
    # Write the sampler client file
    client_file = client_module_path / 'samplerclient.py'
    print(f"Generating {client_file}...")
    with open(client_file, 'w', encoding='utf-8') as f:
        f.write(sampler_code)
    
    # Update __init__.py
    init_file = client_module_path / '__init__.py'
    if not init_file.exists():
        init_file.write_text('')
    
    init_content = init_file.read_text()
    import_line = "from .samplerclient import SamplerClient\n"
    if import_line not in init_content:
        with open(init_file, 'a', encoding='utf-8') as f:
            f.write(import_line)
    
    print("Sampler client generation complete!")


if __name__ == '__main__':
    print("Starting client code generation...\n")
    print("=" * 60)
    
    # Generate processor-based clients
    print("\n[1/3] Generating processor-based clients...")
    parse_interfaces()
    
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

