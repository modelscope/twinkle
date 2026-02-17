# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Markdown Link Validation Tests for Twinkle Documentation

This test suite validates markdown links to ensure ReadTheDocs compatibility.

Usage:
    # Run all tests (skip HTTP validation for speed)
    SKIP_HTTP_LINK_CHECK=true pytest tests/docs/ -v
    
    # Run with HTTP validation (slow, checks all external links)
    pytest tests/docs/ -v
    
    # Check for local relative links (must use GitHub URLs)
    pytest tests/docs/test_markdown_links.py::TestMarkdownLinks::test_no_local_relative_links -v
    
    # Get link statistics
    pytest tests/docs/test_markdown_links.py::TestMarkdownLinks::test_summary_of_links -v -s

Key Requirements:
    - No local relative links (use GitHub URLs for ReadTheDocs compatibility)
    - All GitHub links must use 'main' branch
    - HTTP/HTTPS links should be accessible
"""
import os
import re
import pytest
from pathlib import Path
from typing import List, Tuple
from urllib.parse import urlparse
import requests


DOCS_DIR = Path(__file__).parent.parent.parent / 'docs'
GITHUB_BASE_URL = 'https://github.com/modelscope/twinkle/blob/main'


def find_all_markdown_files(docs_dir: Path) -> List[Path]:
    """Find all markdown files in the docs directory."""
    markdown_files = []
    for root, dirs, files in os.walk(docs_dir):
        for file in files:
            if file.endswith('.md'):
                markdown_files.append(Path(root) / file)
    return markdown_files


def extract_links_from_markdown(file_path: Path) -> List[Tuple[str, str, int]]:
    """
    Extract all markdown links from a file.
    Returns a list of tuples: (link_text, link_url, line_number)
    """
    links = []
    with open(file_path, 'r', encoding='utf-8') as f:
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
    Check if a URL is a local relative link.
    Local relative links should not be used in ReadTheDocs documentation.
    """
    parsed = urlparse(url)
    # If no scheme and not starting with github URL, it's a relative link
    if not parsed.scheme:
        # Exclude anchors (starting with #)
        if url.startswith('#'):
            return False
        return True
    return False


def validate_http_link(url: str, timeout: int = 10) -> Tuple[bool, str]:
    """
    Validate an HTTP/HTTPS link by making a HEAD request.
    Returns (is_valid, error_message)
    """
    try:
        if 'huggingface.co' in url:
            return True, ""
        response = requests.head(url, timeout=timeout, allow_redirects=True)
        if response.status_code < 400:
            return True, ""
        else:
            return False, f"HTTP {response.status_code}"
    except requests.exceptions.Timeout:
        return False, "Timeout"
    except requests.exceptions.RequestException as e:
        return False, str(e)


class TestMarkdownLinks:
    """Test suite for validating markdown links in documentation."""
    
    def test_find_markdown_files(self):
        """Test that we can find markdown files in the docs directory."""
        md_files = find_all_markdown_files(DOCS_DIR)
        assert len(md_files) > 0, "No markdown files found in docs directory"
        print(f"\nFound {len(md_files)} markdown files")
    
    def test_no_local_relative_links(self):
        """
        Test that there are no local relative links in markdown files.
        For ReadTheDocs compatibility, all local file links should use GitHub URLs.
        """
        md_files = find_all_markdown_files(DOCS_DIR)
        violations = []
        
        for md_file in md_files:
            links = extract_links_from_markdown(md_file)
            for text, url, line_num in links:
                if is_local_relative_link(url):
                    relative_path = md_file.relative_to(DOCS_DIR.parent)
                    violations.append({
                        'file': str(relative_path),
                        'line': line_num,
                        'text': text,
                        'url': url,
                        'message': 'Local relative link detected. Use GitHub URL instead.'
                    })
        
        if violations:
            error_msg = "\n\nLocal relative links found (must use GitHub links for ReadTheDocs):\n"
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
    
    @pytest.mark.skipif(
        os.getenv('SKIP_HTTP_LINK_CHECK', 'false').lower() == 'true',
        reason='Skipping HTTP link validation (set SKIP_HTTP_LINK_CHECK=false to enable)'
    )
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
            error_msg = f"\n\nInaccessible HTTP links found ({len(violations)} errors):\n"
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
            with open(md_file, 'r', encoding='utf-8') as f:
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
            error_msg = "\n\nMalformed links found:\n"
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
        
        print(f"\n=== Link Summary ===")
        print(f"Total markdown files: {len(md_files)}")
        print(f"Total links: {total_links}")
        print(f"HTTP/HTTPS links: {http_links}")
        print(f"  - GitHub links: {github_links}")
        print(f"Anchor links (#): {anchor_links}")
        print(f"Local relative links: {relative_links}")
