# Copyright (c) ModelScope Contributors. All rights reserved.
import concurrent.futures
import hashlib
import os
import tempfile
import threading
from concurrent.futures import Future
from pathlib import Path
from requests.exceptions import HTTPError
from typing import Callable, Dict, List, Literal, Optional, Union

from twinkle.utils import requires
from twinkle.utils.parallel import processing_lock
from .model_alias import resolve_model_id_alias

# A thread pool suffices: the pushed payload is network I/O, and a process pool would fork after
# CUDA is up in the importing process.
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)
_futures = {}

large_file_pattern = [
    r'*.bin',
    r'*.safetensors',
    r'*.pth',
    r'*.pt',
    r'*.h5',
    r'*.ckpt',
    r'*.zip',
    r'*.onnx',
    r'*.tar',
    r'*.gz',
]


class HubOperation:

    @staticmethod
    def source_type(resource_name: str):
        resource_name = resource_name or ''
        source_type = 'hf' if resource_name.startswith('hf://') else 'ms'
        if source_type == 'hf' and os.environ.get('TWINKLE_FORBID_HF', '0') != '0':
            # Preventing from hang
            raise ValueError('Using hf as hub backend is not supported.')
        return source_type

    @staticmethod
    def remove_source_type(resource_name: str):
        if not resource_name:
            return resource_name
        parts = resource_name.split('://')
        if len(parts) == 1:
            return parts[0]
        else:
            return parts[-1]

    @classmethod
    def _get_hub_class(cls, resource_name: str) -> type:
        """Get the appropriate Hub class based on resource name prefix.

        Args:
            resource_name: The resource name with optional prefix (hf:// or ms://)

        Returns:
            The Hub class (HFHub or MSHub)
        """
        source = cls.source_type(resource_name)
        return HFHub if source == 'hf' else MSHub

    @classmethod
    def create_model_repo(cls, repo_id: str, token: Optional[str] = None, private: bool = False):
        """Create a model repo on the hub

        Args:
            repo_id: The model id of the hub
            token: The hub token to use
            private: If is a private repo
        """
        hub = cls._get_hub_class(repo_id)
        return hub.create_model_repo(cls.remove_source_type(repo_id), token, private)

    @classmethod
    def push_to_hub(cls,
                    repo_id: str,
                    folder_path: Union[str, Path],
                    path_in_repo: Optional[str] = None,
                    commit_message: Optional[str] = None,
                    commit_description: Optional[str] = None,
                    token: Optional[Union[str, bool]] = None,
                    private: bool = False,
                    revision: Optional[str] = 'master',
                    ignore_patterns: Optional[Union[List[str], str]] = None,
                    **kwargs):
        """Push a model-like folder to the hub

        Args:
            repo_id: The repo id
            folder_path: The local folder path
            path_in_repo: Which remote folder to put the local files in
            commit_message: The commit message of git
            commit_description: The commit description
            token: The hub token
            private: Private hub or not
            revision: The revision to push to
            ignore_patterns: The ignore file patterns
        """
        hub = cls._get_hub_class(repo_id)
        return hub.push_to_hub(
            cls.remove_source_type(repo_id), folder_path, path_in_repo, commit_message, commit_description, token,
            private, revision, ignore_patterns, **kwargs)

    @classmethod
    def upload_checkpoint(cls, repo_id: str, folder_path: str, token: str, report: Callable[..., None]) -> str:
        """Upload a private checkpoint and return the provider's commit ID.

        Hugging Face does not expose per-byte progress through this path.
        """
        if cls.source_type(repo_id) == 'hf':
            return cls.push_to_hub(repo_id, folder_path, token=token, private=True).oid
        return MSHub.upload_checkpoint(cls.remove_source_type(repo_id), folder_path, token, report)

    @classmethod
    def async_push_to_hub(cls,
                          repo_id: str,
                          folder_path: Union[str, Path],
                          path_in_repo: Optional[str] = None,
                          commit_message: Optional[str] = None,
                          commit_description: Optional[str] = None,
                          token: Optional[Union[str, bool]] = None,
                          private: bool = False,
                          revision: Optional[str] = 'master',
                          ignore_patterns: Optional[Union[List[str], str]] = None,
                          **kwargs):
        future: Future = _executor.submit(HubOperation.push_to_hub, repo_id, folder_path, path_in_repo, commit_message,
                                          commit_description, token, private, revision, ignore_patterns, **kwargs)
        replaced = _futures.pop(repo_id, None)
        if replaced is not None and not replaced.done():
            import logging
            logging.getLogger(__name__).warning(
                'A previous push of %r is still in flight and is now dropped by a newer one.', repo_id)
        _futures[repo_id] = future

    @classmethod
    def wait_for(cls, repo_ids: Optional[List[str]] = None) -> Dict[str, str]:
        """Wait for the pushed repos; each waited-on push is popped (it is done from here on)."""
        results = {}
        for repo_id in list(_futures):
            if repo_ids and repo_id not in repo_ids:
                continue
            future = _futures.pop(repo_id)
            try:
                results[repo_id] = future.result()
            except Exception as e:
                results[repo_id] = str(e)
        return results

    @classmethod
    def load_dataset(cls,
                     dataset_id: str,
                     subset_name: str,
                     split: str,
                     streaming: bool = False,
                     revision: Optional[str] = None,
                     **kwargs):
        """Load a dataset from the repo

        Args:
            dataset_id: The dataset id
            subset_name: The subset name of the dataset
            split: The split info
            streaming: Streaming mode
            revision: The revision of the dataset

        Returns:
            The Dataset instance
        """
        hub = cls._get_hub_class(dataset_id)
        return hub.load_dataset(cls.remove_source_type(dataset_id), subset_name, split, streaming, revision, **kwargs)

    @classmethod
    def download_model(cls,
                       model_id_or_path: Optional[str] = None,
                       revision: Optional[str] = None,
                       download_model: bool = True,
                       ignore_patterns: Optional[List[str]] = None,
                       token: Optional[str] = None,
                       **kwargs) -> str:
        """Download model from the hub

        Args:
            model_id_or_path: The model id
            revision: The model revision
            download_model: Whether downloading bin/safetensors files, this is usually useful when only
                using tokenizer
            ignore_patterns: Custom ignore pattern
            token: The hub token
            **kwargs:
                ignore_model: If true, will ignore all `large_file_pattern` files
        Returns:
            The local dir
        """
        model_id_or_path = resolve_model_id_alias(model_id_or_path)
        if not model_id_or_path:
            # ``os.path.exists`` on None would raise a TypeError far from the real cause.
            raise ValueError('download_model needs a model id or local path.')
        if kwargs.pop('ignore_model', False):
            ignore_patterns = set(ignore_patterns or []) | set(large_file_pattern)
        if os.path.exists(model_id_or_path):
            return model_id_or_path
        if os.path.isabs(model_id_or_path) or model_id_or_path.startswith(('./', '../')):
            # A LOCAL path shape that does not exist is never a hub repo id: falling through would
            # send the whole path to the hub (a 404 from modelscope.cn instead of the real
            # diagnosis -- an adapter snapshot read before its directory lands hits exactly this).
            raise FileNotFoundError(
                f'Local model path does not exist: {model_id_or_path}. The path is misspelled, or '
                'the artifact is not written yet (an adapter snapshot still being saved).')
        hub = cls._get_hub_class(model_id_or_path)
        # Every rank of a group reaches this on its own, so without ordering they would all pull the
        # same repo into the same cache dir at once. The key covers the patterns too, so a partial
        # fetch (weightless, or a single file) never satisfies a caller that wants more. md5 rather
        # than hash(), which is salted per process and would give every rank a different key.
        scope = str(sorted(ignore_patterns or [])) + str(kwargs.get('allow_patterns') or '')
        scope = hashlib.md5(scope.encode('utf-8')).hexdigest()[:8]
        with processing_lock(f'download_model:{model_id_or_path}:{scope}', sticky=True):
            return hub.download_model(
                model_id_or_path=cls.remove_source_type(model_id_or_path),
                revision=revision,
                ignore_patterns=ignore_patterns,
                token=token,
                **kwargs)



class MSHub(HubOperation):
    @staticmethod
    def _ensure_configuration(folder_path):
        config = Path(folder_path) / 'configuration.json'
        if not config.exists():
            config.write_text(
                '{"framework": "pytorch", "task": "text-generation", "allow_remote": true}', encoding='utf-8')

    @staticmethod
    def upload_checkpoint(repo_id: str, folder_path: str, token: str, report: Callable[..., None]) -> str:
        requires('modelscope-hub')
        from modelscope_hub import HubApi
        from modelscope_hub.errors import AlreadyExistsError

        api = HubApi(token=token)
        try:
            api.create_repo(repo_id, 'model', visibility='private')
        except AlreadyExistsError:
            pass

        MSHub._ensure_configuration(folder_path)
        stages = {
            'upload_progress': 'uploading',
            'batch_committed': 'batch_committed',
            'recovery_committed': 'batch_committed',
            'batch_failed': 'batch_failed',
        }
        uploaded = committed = inline = 0
        lock = threading.Lock()

        def progress(event):
            nonlocal uploaded, committed, inline
            with lock:
                uploaded = max(uploaded, event.get('uploaded_bytes', 0))
                committed = max(committed, event['committed_bytes'])
                inline += event.get('batch_inline_bytes', 0)
                report(stage=stages[event['event']], uploaded_bytes=uploaded + inline,
                       total_bytes=event['total_bytes'], committed_bytes=committed)

        # A fresh upload lets the Hub deduplicate bytes and still return its exact commit ID.
        result = api.upload_folder(
            repo_id, 'model', folder_path, use_cache=False, disable_tqdm=True, progress_callback=progress)
        commit = result[-1] if isinstance(result, list) and result else result
        revision = commit.get('commit_id') if isinstance(commit, dict) else None
        if not revision:
            raise RuntimeError(f'ModelScope did not return a commit ID for {repo_id}.')
        return revision

    @classmethod
    def try_login(cls, token: Optional[str] = None) -> bool:
        requires('modelscope')
        from modelscope.hub.api import HubApi
        if token is None:
            token = os.environ.get('MODELSCOPE_API_TOKEN')
        if token:
            api = HubApi()
            api.login(token)
            return True
        return False

    @classmethod
    def create_model_repo(cls, repo_id: str, token: Optional[str] = None, private: bool = False) -> str:
        requires('modelscope')
        from modelscope.hub.api import HubApi, ModelScopeConfig
        from modelscope.hub.constants import ModelVisibility
        assert repo_id is not None, 'Please enter a valid hub_model_id'

        if not cls.try_login(token):
            raise ValueError('Please specify a token by `--hub_token` or `MODELSCOPE_API_TOKEN=xxx`')
        visibility = ModelVisibility.PRIVATE if private else ModelVisibility.PUBLIC
        api = HubApi()
        if '/' not in repo_id:
            user_name = ModelScopeConfig.get_user_info()[0]
            assert isinstance(user_name, str)
        try:
            api.create_model(repo_id, visibility=visibility)
        except HTTPError:
            # The remote repository has been created
            pass

        with tempfile.TemporaryDirectory() as temp_cache_dir:
            from modelscope.hub.repository import Repository
            repo = Repository(temp_cache_dir, repo_id)
            cls.add_patterns_to_gitattributes(repo, ['*.safetensors', '*.bin', '*.pt'])
            # Add 'runs/' to .gitignore, ignore tensorboard files
            cls.add_patterns_to_gitignore(repo, ['runs/', 'images/'])
            cls.add_patterns_to_file(
                repo,
                'configuration.json', ['{"framework": "pytorch", "task": "text-generation", "allow_remote": true}'],
                ignore_push_error=True)
            # Add '*.sagemaker' to .gitignore if using SageMaker
            if os.environ.get('SM_TRAINING_ENV'):
                cls.add_patterns_to_gitignore(repo, ['*.sagemaker-uploading', '*.sagemaker-uploaded'],
                                              'Add `*.sagemaker` patterns to .gitignore')
        return repo_id

    @classmethod
    def push_to_hub(cls,
                    repo_id: str,
                    folder_path: Union[str, Path],
                    path_in_repo: Optional[str] = None,
                    commit_message: Optional[str] = None,
                    commit_description: Optional[str] = None,
                    token: Optional[Union[str, bool]] = None,
                    private: bool = False,
                    revision: Optional[str] = 'master',
                    ignore_patterns: Optional[Union[List[str], str]] = None,
                    **kwargs):
        requires('modelscope')
        # ModelScope creates the repository as part of the upload. Use that API
        # directly so uploading does not require a separate Git checkout.
        from modelscope import push_to_hub
        commit_message = commit_message or 'Upload folder using api'
        if commit_description:
            commit_message = commit_message + '\n' + commit_description
        cls._ensure_configuration(folder_path)
        if ignore_patterns:
            ignore_patterns = [p for p in ignore_patterns if p != '_*']
        if path_in_repo:
            # We don't support part submit for now
            path_in_repo = os.path.basename(folder_path)
            folder_path = os.path.dirname(folder_path)
            ignore_patterns = []
        if revision is None or revision == 'main':
            revision = 'master'
        try:
            result = push_to_hub(
                repo_id,
                folder_path,
                token=token,
                private=private,
                commit_message=commit_message,
                ignore_file_pattern=ignore_patterns,
                revision=revision,
                tag=path_in_repo)
        except Exception as exc:
            raise RuntimeError(f'ModelScope push_to_hub raised an exception '
                               f'(repo_id={repo_id!r}, folder_path={folder_path!r}): {exc}') from exc
        if not result:
            raise RuntimeError(f'ModelScope push_to_hub returned a falsy result '
                               f'(repo_id={repo_id!r}, folder_path={folder_path!r}). '
                               f'This usually indicates an invalid/expired token or insufficient write permission.')

    @classmethod
    def load_dataset(cls,
                     dataset_id: str,
                     subset_name: str,
                     split: str,
                     streaming: bool = False,
                     revision: Optional[str] = None,
                     download_mode: Literal['force_redownload', 'reuse_dataset_if_exists'] = 'reuse_dataset_if_exists',
                     token: Optional[str] = None,
                     **kwargs):
        requires('modelscope')
        from modelscope import MsDataset
        cls.try_login(token)
        if revision is None or revision == 'main':
            revision = 'master'
        load_kwargs = {'trust_remote_code': kwargs.get('trust_remote_code', True)}
        return MsDataset.load(
            dataset_id,
            subset_name=subset_name,
            split=split,
            version=revision,
            download_mode=download_mode,  # noqa
            use_streaming=streaming,
            **load_kwargs,
        )

    @classmethod
    def download_model(cls,
                       model_id_or_path: Optional[str] = None,
                       revision: Optional[str] = None,
                       ignore_patterns: Optional[List[str]] = None,
                       token: Optional[str] = None,
                       **kwargs):
        requires('modelscope')
        cls.try_login(token)
        if revision is None or revision == 'main':
            revision = 'master'
        import inspect
        from modelscope import snapshot_download

        # Build download arguments
        download_kwargs = {
            'model_id': model_id_or_path,
            'revision': revision,
            'ignore_patterns': ignore_patterns,
            **kwargs
        }

        # Add token parameter only if supported by the function signature
        if token is not None:
            sig = inspect.signature(snapshot_download)
            if 'token' in sig.parameters:
                download_kwargs['token'] = token
            else:
                print('Token parameter is not supported by current modelscope version. '
                      'Please upgrade to modelscope >= 1.34.0 for token-based authentication.')

        return snapshot_download(**download_kwargs)

    @staticmethod
    def add_patterns_to_file(repo,
                             file_name: str,
                             patterns: List[str],
                             commit_message: Optional[str] = None,
                             ignore_push_error=False) -> None:
        if isinstance(patterns, str):
            patterns = [patterns]
        if commit_message is None:
            commit_message = f'Add `{patterns[0]}` patterns to {file_name}'

        # Get current file content
        repo_dir = repo.model_dir
        file_path = os.path.join(repo_dir, file_name)
        if os.path.exists(file_path):
            with open(file_path, encoding='utf-8') as f:
                current_content = f.read()
        else:
            current_content = ''
        # Add the patterns to file
        content = current_content
        for pattern in patterns:
            if pattern not in content:
                if len(content) > 0 and not content.endswith('\n'):
                    content += '\n'
                content += f'{pattern}\n'

        # Write the file if it has changed
        if content != current_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        try:
            repo.push(commit_message)
        except Exception as e:
            if ignore_push_error:
                pass
            else:
                raise e

    @staticmethod
    def add_patterns_to_gitignore(repo, patterns: List[str], commit_message: Optional[str] = None) -> None:
        MSHub.add_patterns_to_file(repo, '.gitignore', patterns, commit_message, ignore_push_error=True)

    @staticmethod
    def add_patterns_to_gitattributes(repo, patterns: List[str], commit_message: Optional[str] = None) -> None:
        new_patterns = []
        suffix = 'filter=lfs diff=lfs merge=lfs -text'
        for pattern in patterns:
            if suffix not in pattern:
                pattern = f'{pattern} {suffix}'
            new_patterns.append(pattern)
        file_name = '.gitattributes'
        if commit_message is None:
            commit_message = f'Add `{patterns[0]}` patterns to {file_name}'
        MSHub.add_patterns_to_file(repo, file_name, new_patterns, commit_message, ignore_push_error=True)


class HFHub(HubOperation):

    @classmethod
    def create_model_repo(cls, repo_id: str, token: Optional[str] = None, private: bool = False) -> str:
        requires('huggingface_hub')
        from huggingface_hub.hf_api import api
        # exist_ok: a re-run export pushes onto a repo it created the first time; the MS path
        # tolerates the same through its HTTPError branch below.
        return api.create_repo(repo_id, token=token, private=private, exist_ok=True)

    @classmethod
    def push_to_hub(cls,
                    repo_id: str,
                    folder_path: Union[str, Path],
                    path_in_repo: Optional[str] = None,
                    commit_message: Optional[str] = None,
                    commit_description: Optional[str] = None,
                    token: Optional[Union[str, bool]] = None,
                    private: bool = False,
                    revision: Optional[str] = 'master',
                    ignore_patterns: Optional[Union[List[str], str]] = None,
                    **kwargs):
        requires('huggingface_hub')
        from huggingface_hub.hf_api import api
        cls.create_model_repo(repo_id, token, private)
        if revision is None or revision == 'master':
            revision = 'main'
        return api.upload_folder(
            repo_id=repo_id,
            folder_path=folder_path,
            path_in_repo=path_in_repo,
            commit_message=commit_message,
            commit_description=commit_description,
            token=token,
            revision=revision,
            ignore_patterns=ignore_patterns,
            **kwargs)

    @classmethod
    def load_dataset(cls,
                     dataset_id: str,
                     subset_name: str,
                     split: str,
                     streaming: bool = False,
                     revision: Optional[str] = None,
                     download_mode: Literal['force_redownload', 'reuse_dataset_if_exists'] = 'reuse_dataset_if_exists',
                     num_proc: Optional[int] = None,
                     **kwargs):
        requires('huggingface_hub')
        requires('datasets')
        from datasets import load_dataset
        if revision is None or revision == 'master':
            revision = 'main'
        trust_remote_code = kwargs.get('trust_remote_code', True)
        return load_dataset(
            dataset_id,
            name=subset_name,
            split=split,
            streaming=streaming,
            revision=revision,
            download_mode=download_mode,
            trust_remote_code=trust_remote_code,
            num_proc=num_proc)

    @classmethod
    def download_model(cls,
                       model_id_or_path: Optional[str] = None,
                       revision: Optional[str] = None,
                       ignore_patterns: Optional[List[str]] = None,
                       token: Optional[str] = None,
                       **kwargs):
        if revision is None or revision == 'master':
            revision = 'main'
        from huggingface_hub import snapshot_download
        return snapshot_download(
            repo_id=model_id_or_path,
            repo_type='model',
            revision=revision,
            ignore_patterns=ignore_patterns,
            token=token,
            **kwargs)
