# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import tempfile
from concurrent.futures import Future
from contextlib import contextmanager
from pathlib import Path
from typing import List, Literal, Optional, Union, Dict
import concurrent.futures

from requests.exceptions import HTTPError
from ..utils import requires

_executor = concurrent.futures.ProcessPoolExecutor(max_workers=8)
_futures = {}


large_file_pattern = [
    r'\w+\.bin',
    r'\w+\.safetensors',
    r'\w+\.pth',
    r'\w+\.pt',
    r'\w+\.h5',
    r'\w+\.ckpt',
    r'\w+\.zip',
    r'\w+\.onnx',
    r'\w+\.tar',
    r'\w+\.gz',
]


class HubOperation:

    @classmethod
    @contextmanager
    def patch_hub(cls):
        yield

    @staticmethod
    def source_type(resource_name: str):
        resource_name = resource_name or ''
        if resource_name.startswith('hf://'):
            source_type = 'hf'
        elif resource_name.startswith('ms://'):
            source_type = 'ms'
        else:
            source_type = 'ms'
        if source_type == 'hf' and os.environ.get('TWINKLE_FORBID_HF', '0') != '0':
            # Preventing from hang
            raise ValueError(f'Using hf as hub backend is not supported.')
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
        if source == 'hf':
            return HFHub
        elif source == 'ms':
            return MSHub
        else:
            raise NotImplementedError(f'Unknown source type: {source}')

    @classmethod
    def try_login(cls, token: Optional[str] = None) -> bool:
        """Try to log in to the hub

        Args:
            token: The hub token to use

        Returns:
            bool: Whether login is successful
        """
        hub = cls._get_hub_class(token)
        return hub.try_login(cls.remove_source_type(token))

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
            cls.remove_source_type(repo_id), folder_path, path_in_repo, commit_message,
            commit_description, token, private, revision, ignore_patterns, **kwargs)

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
        future: Future = _executor.submit(HubOperation.push_to_hub, repo_id, folder_path, path_in_repo,
                                          commit_message, commit_description, token, private,
                                          revision, ignore_patterns, **kwargs)
        _futures[repo_id] = future

    @classmethod
    def wait_for(cls, repo_ids: Optional[List[str]] = None) -> Dict[str, str]:
        results = {}
        for repo_id, future in _futures.items():
            future: Future
            if not repo_ids or repo_id in repo_ids:
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
                     revision: Optional[str] = None):
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
        return hub.load_dataset(cls.remove_source_type(dataset_id), subset_name, split, streaming, revision)

    @classmethod
    def download_model(cls,
                       model_id_or_path: Optional[str] = None,
                       revision: Optional[str] = None,
                       download_model: bool = True,
                       ignore_patterns: Optional[List[str]] = [],
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
        if kwargs.pop('ignore_model', False):
            ignore_patterns = set(ignore_patterns or []
                                  ) | set(large_file_pattern)
        if os.path.exists(model_id_or_path):
            return model_id_or_path
        hub = cls._get_hub_class(model_id_or_path)
        return hub.download_model(
            model_id_or_path=cls.remove_source_type(model_id_or_path),
            revision=revision,
            ignore_patterns=ignore_patterns,
            token=token,
            **kwargs)

    @classmethod
    def download_file(cls,
                      repo_id: str,
                      repo_type: str = 'model',
                      allow_patterns: Optional[Union[List[str], str]] = None,
                      token: Optional[str] = None,
                      **kwargs) -> str:
        """Download specific files from the hub

        Args:
            repo_id: The repository id
            repo_type: The type of repository, default is 'model'
            allow_patterns: Patterns to filter which files to download
            token: The hub token
            **kwargs: Additional arguments passed to the download function

        Returns:
            The local directory path containing downloaded files
        """
        hub = cls._get_hub_class(repo_id)
        return hub.download_file(
            repo_id=cls.remove_source_type(repo_id),
            repo_type=repo_type,
            allow_patterns=allow_patterns,
            token=token,
            **kwargs)


class MSHub(HubOperation):
    ms_token = None

    @staticmethod
    def create_repo(repo_id: str,
                    *,
                    token: Optional[Union[str, bool]] = None,
                    private: bool = False,
                    **kwargs) -> 'modelscope.utils.repo_utils.RepoUrl':
        """
        Create a new repository on the hub.

        Args:
            repo_id: The ID of the repository to create.
            token: The authentication token to use.
            private: Whether the repository should be private.
            **kwargs: Additional arguments.

        Returns:
            RepoUrl: The URL of the created repository.
        """
        requires('modelscope')
        hub_model_id = MSHub.create_model_repo(repo_id, token, private)
        from modelscope.utils.repo_utils import RepoUrl
        return RepoUrl(url=hub_model_id, )

    @staticmethod
    def upload_folder(
        *,
        repo_id: str,
        folder_path: Union[str, Path],
        path_in_repo: Optional[str] = None,
        commit_message: Optional[str] = None,
        commit_description: Optional[str] = None,
        token: Optional[Union[str, bool]] = None,
        revision: Optional[str] = 'master',
        ignore_patterns: Optional[Union[List[str], str]] = None,
        **kwargs,
    ):
        requires('modelscope')
        from modelscope.utils.repo_utils import CommitInfo
        MSHub.push_to_hub(repo_id, folder_path, path_in_repo, commit_message, commit_description, token, True, revision,
                          ignore_patterns)
        return CommitInfo(
            commit_url=f'https://www.modelscope.cn/models/{repo_id}/files',
            commit_message=commit_message,
            commit_description=commit_description,
            oid='',
        )

    @classmethod
    def try_login(cls, token: Optional[str] = None) -> bool:
        requires('modelscope')
        from modelscope import HubApi
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
        from modelscope import HubApi
        from modelscope.hub.api import ModelScopeConfig
        from modelscope.hub.constants import ModelVisibility
        assert repo_id is not None, 'Please enter a valid hub_model_id'

        if not cls.try_login(token):
            raise ValueError(
                'Please specify a token by `--hub_token` or `MODELSCOPE_API_TOKEN=xxx`')
        cls.ms_token = token
        visibility = ModelVisibility.PRIVATE if private else ModelVisibility.PUBLIC
        api = HubApi()
        if '/' not in repo_id:
            user_name = ModelScopeConfig.get_user_info()[0]
            assert isinstance(user_name, str)
        try:
            api.create_model(repo_id, visibility)
        except HTTPError:
            # The remote repository has been created
            pass

        with tempfile.TemporaryDirectory() as temp_cache_dir:
            from modelscope.hub.repository import Repository
            repo = Repository(temp_cache_dir, repo_id)
            cls.add_patterns_to_gitattributes(
                repo, ['*.safetensors', '*.bin', '*.pt'])
            # Add 'runs/' to .gitignore, ignore tensorboard files
            cls.add_patterns_to_gitignore(repo, ['runs/', 'images/'])
            cls.add_patterns_to_file(
                repo,
                'configuration.json', [
                    '{"framework": "pytorch", "task": "text-generation", "allow_remote": true}'],
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
        cls.create_model_repo(repo_id, token, private)
        from modelscope import push_to_hub
        commit_message = commit_message or 'Upload folder using api'
        if commit_description:
            commit_message = commit_message + '\n' + commit_description
        if not os.path.exists(os.path.join(folder_path, 'configuration.json')):
            with open(os.path.join(folder_path, 'configuration.json'), 'w', encoding='utf-8') as f:
                f.write(
                    '{"framework": "pytorch", "task": "text-generation", "allow_remote": true}')
        if ignore_patterns:
            ignore_patterns = [p for p in ignore_patterns if p != '_*']
        if path_in_repo:
            # We don't support part submit for now
            path_in_repo = os.path.basename(folder_path)
            folder_path = os.path.dirname(folder_path)
            ignore_patterns = []
        if revision is None or revision == 'main':
            revision = 'master'
        return push_to_hub(
                repo_id,
                folder_path,
                token or cls.ms_token,
                private,
                commit_message=commit_message,
                ignore_file_pattern=ignore_patterns,
                revision=revision,
                tag=path_in_repo)

    @classmethod
    def load_dataset(cls,
                     dataset_id: str,
                     subset_name: str,
                     split: str,
                     streaming: bool = False,
                     revision: Optional[str] = None,
                     download_mode: Literal['force_redownload',
                                            'reuse_dataset_if_exists'] = 'reuse_dataset_if_exists',
                     token: Optional[str] = None,
                     **kwargs):
        requires('modelscope')
        from modelscope import MsDataset
        cls.try_login(token)
        if revision is None or revision == 'main':
            revision = 'master'
        load_kwargs = {'trust_remote_code': True}
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
        from modelscope import snapshot_download
        import inspect
        
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
                print(
                    'Token parameter is not supported by current modelscope version. '
                    'Please upgrade to modelscope >= 1.34.0 for token-based authentication.'
                )
        
        return snapshot_download(**download_kwargs)

    @classmethod
    def download_file(cls,
                      repo_id: str,
                      repo_type: str = 'model',
                      allow_patterns: Optional[Union[List[str], str]] = None,
                      token: Optional[str] = None,
                      **kwargs) -> str:
        """Download specific files from ModelScope hub

        Args:
            repo_id: The repository id
            repo_type: The type of repository, default is 'model'
            allow_patterns: Patterns to filter which files to download
            token: The hub token
            **kwargs: Additional arguments passed to _snapshot_download

        Returns:
            The local directory path containing downloaded files
        """
        requires('modelscope')
        cls.try_login(token)
        from modelscope.hub.snapshot_download import _snapshot_download
        import inspect

        # Build download arguments
        download_kwargs = {
            'repo_id': repo_id,
            'repo_type': repo_type,
            'allow_patterns': allow_patterns,
            **kwargs
        }

        # Add token parameter only if supported by the function signature
        if token is not None:
            sig = inspect.signature(_snapshot_download)
            if 'token' in sig.parameters:
                download_kwargs['token'] = token
            else:
                print(
                    'Token parameter is not supported by current modelscope version. '
                    'Please upgrade to modelscope >= 1.34.0 for token-based authentication.'
                )

        return _snapshot_download(**download_kwargs)

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
            with open(file_path, 'r', encoding='utf-8') as f:
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
        MSHub.add_patterns_to_file(
            repo, '.gitignore', patterns, commit_message, ignore_push_error=True)

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
        MSHub.add_patterns_to_file(
            repo, file_name, new_patterns, commit_message, ignore_push_error=True)


class HFHub(HubOperation):

    @classmethod
    def try_login(cls, token: Optional[str] = None) -> bool:
        pass

    @classmethod
    def create_model_repo(cls, repo_id: str, token: Optional[str] = None, private: bool = False) -> str:
        requires('huggingface_hub')
        from huggingface_hub.hf_api import api
        return api.create_repo(repo_id, token=token, private=private)

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
                     download_mode: Literal['force_redownload',
                                            'reuse_dataset_if_exists'] = 'reuse_dataset_if_exists',
                     num_proc: Optional[int] = None,
                     **kwargs):
        requires('huggingface_hub')
        requires('datasets')
        from datasets import load_dataset
        if revision is None or revision == 'master':
            revision = 'main'
        return load_dataset(
            dataset_id,
            name=subset_name,
            split=split,
            streaming=streaming,
            revision=revision,
            download_mode=download_mode,
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
            **kwargs
        )

    @classmethod
    def download_file(cls,
                      repo_id: str,
                      repo_type: str = 'model',
                      allow_patterns: Optional[Union[List[str], str]] = None,
                      token: Optional[str] = None,
                      **kwargs) -> str:
        """Download specific files from HuggingFace hub

        Args:
            repo_id: The repository id
            repo_type: The type of repository, default is 'model'
            allow_patterns: Patterns to filter which files to download
            token: The hub token
            **kwargs: Additional arguments passed to snapshot_download

        Returns:
            The local directory path containing downloaded files
        """
        requires('huggingface_hub')
        from huggingface_hub import snapshot_download
        return snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            allow_patterns=allow_patterns,
            token=token,
            **kwargs
        )
