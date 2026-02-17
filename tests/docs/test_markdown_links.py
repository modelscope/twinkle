# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import pytest
import re
import requests
from pathlib import Path
from typing import List, Tuple
from urllib.parse import urlparse

DOCS_DIR = Path(__file__).parent.parent.parent / 'docs'
GITHUB_BASE_URL = 'https://github.com/modelscope/twinkle/blob/main'

# Files and folders to skip during validation (relative to DOCS_DIR)
# Can be set via environment variable: SKIP_DOC_PATHS="build,_build,temp.md"
# Paths should be relative to docs/ directory, not including 'docs/' prefix
SKIP_PATHS = ['source_en/Usage Guide/NPU-Support.md', 'source_zh/使用指引/NPU的支持.md']


def should_skip_path(path: Path, docs_dir: Path, skip_paths: List[str]) -> bool:
    """
    Check if a path should be skipped based on skip_paths configuration.

    Args:
        path: Path to check (can be file or directory)
        docs_dir: Root docs directory
        skip_paths: List of paths to skip (relative to docs_dir)

    Returns:
        True if path should be skipped, False otherwise
    """
    if not skip_paths:
        return False

    try:
        rel_path = path.relative_to(docs_dir)
    except ValueError:
        return False

    rel_path_str = str(rel_path)

    for skip_path in skip_paths:
        skip_path = skip_path.strip()
        if not skip_path:
            continue

        # Check if it's an exact match or if the path is under a skipped directory
        if rel_path_str == skip_path or rel_path_str.startswith(skip_path + '/'):
            return True

    return False


def find_all_markdown_files(docs_dir: Path, skip_paths: List[str] = None) -> List[Path]:
    """
    Find all markdown files in the docs directory.

    Args:
        docs_dir: Root directory to search for markdown files
        skip_paths: List of paths (files or folders) to skip relative to docs_dir
                   Example: ['build', '_build', 'temp/draft.md']

    Returns:
        List of Path objects for all markdown files found
    """
    if skip_paths is None:
        skip_paths = SKIP_PATHS

    markdown_files = []
    for root, dirs, files in os.walk(docs_dir):
        root_path = Path(root)

        # Check if current directory should be skipped
        if should_skip_path(root_path, docs_dir, skip_paths):
            # Clear dirs to prevent os.walk from descending into subdirectories
            dirs[:] = []
            continue

        # Process files in current directory
        for file in files:
            if file.endswith('.md'):
                file_path = root_path / file
                # Check if specific file should be skipped
                if not should_skip_path(file_path, docs_dir, skip_paths):
                    markdown_files.append(file_path)

    return markdown_files


def extract_links_from_markdown(file_path: Path) -> List[Tuple[str, str, int]]:
    """
    Extract all markdown links from a file.
    Returns a list of tuples: (link_text, link_url, line_number)
    """
    links = []
    with open(file_path, encoding='utf-8') as f:
        content = f.readlines()

    # Pattern to match markdown links: [text](url)
    link_pattern = re.compile(r'\[([^\]]+)\]\(([^\)]+)\)')

    for line_num, line in enumerate(content, start=1):
        matches = link_pattern.findall(line)
        for text, url in matches:
            links.append((text, url, line_num))

    return links


def is_http_link(url: str) -> bool:
    """Check if a URL is an HTTP/HTTPS link."""
    parsed = urlparse(url)
    return parsed.scheme in ('http', 'https')


def is_local_relative_link(url: str) -> bool:
    """
    Check if a URL is a local relative link (not HTTP/HTTPS).
    """
    parsed = urlparse(url)
    # If no scheme and not starting with github URL, it's a relative link
    if not parsed.scheme:
        # Exclude anchors (starting with #)
        if url.startswith('#'):
            return False
        return True
    return False


def is_link_outside_docs(url: str, current_file: Path, docs_dir: Path) -> bool:
    """
    Check if a local relative link points to a file outside the docs directory.

    Args:
        url: The link URL (relative path, may include fragment like 'path/file.md#anchor')
        current_file: The markdown file containing this link
        docs_dir: Root docs directory

    Returns:
        True if link points outside docs/, False if within docs/ or cannot determine
    """
    if not is_local_relative_link(url):
        return False

    # Handle URL fragments by splitting at '#'
    url_path = url.split('#')[0]
    if not url_path:
        # This is an anchor-only link like '#section', not a file link
        return False

    # Resolve the target path relative to the current file's directory
    current_dir = current_file.parent
    try:
        # Resolve the target path
        target_path = (current_dir / url_path).resolve()

        # Check if target is within docs directory
        try:
            target_path.relative_to(docs_dir.resolve())
            # Target is within docs directory
            return False
        except ValueError:
            # Target is outside docs directory
            return True
    except Exception:
        # If we can't resolve the path, assume it's problematic
        return True


def validate_http_link(url: str, timeout: int = 10) -> Tuple[bool, str]:
    """
    Validate an HTTP/HTTPS link by making a HEAD request.
    Returns (is_valid, error_message)
    """
    try:
        if 'huggingface.co' in url:
            return True, ''
        response = requests.head(url, timeout=timeout, allow_redirects=True)
        if response.status_code < 400:
            return True, ''
        else:
            return False, f'HTTP {response.status_code}'
    except requests.exceptions.Timeout:
        return False, 'Timeout'
    except requests.exceptions.RequestException as e:
        return False, str(e)


class TestMarkdownLinks:
    """Test suite for validating markdown links in documentation."""

    def test_find_markdown_files(self):
        """Test that we can find markdown files in the docs directory."""
        md_files = find_all_markdown_files(DOCS_DIR)
        assert len(md_files) > 0, 'No markdown files found in docs directory'
        print(f'\nFound {len(md_files)} markdown files')

    def test_no_local_relative_links(self):
        """
        Test that local relative links pointing outside docs/ use GitHub URLs.
        Links within docs/ directory can use relative paths (for ReadTheDocs compatibility).
        Links to other directories (cookbook/, src/, etc.) must use GitHub URLs.
        """
        md_files = find_all_markdown_files(DOCS_DIR)
        violations = []

        for md_file in md_files:
            links = extract_links_from_markdown(md_file)
            for text, url, line_num in links:
                # Check if this is a local relative link pointing outside docs/
                if is_link_outside_docs(url, md_file, DOCS_DIR):
                    relative_path = md_file.relative_to(DOCS_DIR.parent)
                    violations.append({
                        'file': str(relative_path),
                        'line': line_num,
                        'text': text,
                        'url': url,
                        'message': 'Local relative link to file outside docs/. Use GitHub URL instead.'
                    })

        if violations:
            error_msg = '\n\nLocal relative links to files outside docs/ found (must use GitHub links):\n'
            for v in violations:
                error_msg += f"\n  File: {v['file']}:{v['line']}\n"
                error_msg += f"  Link: [{v['text']}]({v['url']})\n"
                error_msg += f"  Message: {v['message']}\n"
            pytest.fail(error_msg)

    def test_github_links_use_main_branch(self):
        """
        Test that all GitHub links use the 'main' branch.
        """
        md_files = find_all_markdown_files(DOCS_DIR)
        violations = []

        github_pattern = re.compile(r'https://github\.com/[^/]+/[^/]+/blob/([^/]+)/')

        for md_file in md_files:
            links = extract_links_from_markdown(md_file)
            for text, url, line_num in links:
                match = github_pattern.search(url)
                if match:
                    branch = match.group(1)
                    if branch != 'main':
                        relative_path = md_file.relative_to(DOCS_DIR.parent)
                        violations.append({
                            'file': str(relative_path),
                            'line': line_num,
                            'text': text,
                            'url': url,
                            'branch': branch,
                            'message': f'GitHub link uses branch "{branch}" instead of "main"'
                        })

        if violations:
            error_msg = "\n\nGitHub links not using 'main' branch:\n"
            for v in violations:
                error_msg += f"\n  File: {v['file']}:{v['line']}\n"
                error_msg += f"  Link: [{v['text']}]({v['url']})\n"
                error_msg += f"  Message: {v['message']}\n"
            pytest.fail(error_msg)

    def test_http_links_are_accessible(self):
        """
        Test that all HTTP/HTTPS links are accessible.
        This test can be slow, so it can be skipped by setting SKIP_HTTP_LINK_CHECK=true.
        """
        md_files = find_all_markdown_files(DOCS_DIR)
        violations = []
        checked_urls = {}  # Cache to avoid checking the same URL multiple times

        for md_file in md_files:
            links = extract_links_from_markdown(md_file)
            for text, url, line_num in links:
                if is_http_link(url):
                    # Check cache first
                    if url in checked_urls:
                        is_valid, error = checked_urls[url]
                    else:
                        is_valid, error = validate_http_link(url)
                        checked_urls[url] = (is_valid, error)

                    if not is_valid:
                        relative_path = md_file.relative_to(DOCS_DIR.parent)
                        violations.append({
                            'file': str(relative_path),
                            'line': line_num,
                            'text': text,
                            'url': url,
                            'error': error
                        })

        if violations:
            error_msg = f'\n\nInaccessible HTTP links found ({len(violations)} errors):\n'
            for v in violations:
                error_msg += f"\n  File: {v['file']}:{v['line']}\n"
                error_msg += f"  Link: [{v['text']}]({v['url']})\n"
                error_msg += f"  Error: {v['error']}\n"
            pytest.fail(error_msg)

    def test_link_format_is_valid(self):
        """
        Test that all links follow valid markdown link format.
        This test checks for common malformed link patterns within the same line.
        """
        md_files = find_all_markdown_files(DOCS_DIR)
        violations = []

        for md_file in md_files:
            with open(md_file, encoding='utf-8') as f:
                lines = f.readlines()

            for line_num, line in enumerate(lines, start=1):
                # Check for spaces after ]( or before ) within a link on the same line
                # Pattern: ](  with space after opening paren
                if re.search(r'\]\(\s+', line):
                    # Make sure it's not in a code block (lines with ```)
                    if not line.strip().startswith('```'):
                        relative_path = md_file.relative_to(DOCS_DIR.parent)
                        violations.append({
                            'file': str(relative_path),
                            'line': line_num,
                            'line_content': line.strip()[:80],
                            'message': 'Space after opening parenthesis in markdown link: ]( '
                        })

        if violations:
            error_msg = '\n\nMalformed links found:\n'
            for v in violations:
                error_msg += f"\n  File: {v['file']}:{v['line']}\n"
                error_msg += f"  Line: {v['line_content']}\n"
                error_msg += f"  Message: {v['message']}\n"
            pytest.fail(error_msg)

    def test_summary_of_links(self):
        """
        Generate a summary of all links found in the documentation.
        This is not a validation test, just informational.
        """
        md_files = find_all_markdown_files(DOCS_DIR)
        total_links = 0
        http_links = 0
        github_links = 0
        relative_links = 0
        anchor_links = 0

        for md_file in md_files:
            links = extract_links_from_markdown(md_file)
            for text, url, line_num in links:
                total_links += 1
                if url.startswith('#'):
                    anchor_links += 1
                elif is_http_link(url):
                    http_links += 1
                    if 'github.com' in url:
                        github_links += 1
                elif is_local_relative_link(url):
                    relative_links += 1

        print('\n=== Link Summary ===')
        print(f'Total markdown files: {len(md_files)}')
        print(f'Total links: {total_links}')
        print(f'HTTP/HTTPS links: {http_links}')
        print(f'  - GitHub links: {github_links}')
        print(f'Anchor links (#): {anchor_links}')
        print(f'Local relative links: {relative_links}')

    def test_readme_files_http_links(self):
        """
        Test that HTTP/HTTPS links in README files are accessible.
        README files are not on ReadTheDocs, so they can use local project paths.
        This test only validates HTTP/HTTPS links.
        """
        project_root = DOCS_DIR.parent
        readme_files = [project_root / 'README.md', project_root / 'README_ZH.md']

        violations = []
        checked_urls = {}  # Cache to avoid checking the same URL multiple times

        for readme_file in readme_files:
            if not readme_file.exists():
                continue

            links = extract_links_from_markdown(readme_file)
            for text, url, line_num in links:
                # Only check HTTP/HTTPS links
                if is_http_link(url):
                    # Check cache first
                    if url in checked_urls:
                        is_valid, error = checked_urls[url]
                    else:
                        is_valid, error = validate_http_link(url)
                        checked_urls[url] = (is_valid, error)

                    if not is_valid:
                        relative_path = readme_file.relative_to(project_root)
                        violations.append({
                            'file': str(relative_path),
                            'line': line_num,
                            'text': text,
                            'url': url,
                            'error': error
                        })

        if violations:
            error_msg = f'\n\nInaccessible HTTP links found in README files ({len(violations)} errors):\n'
            for v in violations:
                error_msg += f"\n  File: {v['file']}:{v['line']}\n"
                error_msg += f"  Link: [{v['text']}]({v['url']})\n"
                error_msg += f"  Error: {v['error']}\n"
            pytest.fail(error_msg)
