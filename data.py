#!/usr/bin/env python3
"""
NSIDC Data Downloader
Downloads all files from NOAA/NSIDC directory listings.

Usage:
    python nsidc_downloader.py [URL] [OUTPUT_DIR]
    
Example:
    python nsidc_downloader.py https://noaadata.apps.nsidc.org/NOAA/G02135/south/daily/images/ ./downloaded_data
"""

import os
import sys
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import argparse


class NSIDCDownloader:
    def __init__(self, base_url, output_dir, max_workers=5, delay=0.5):
        """
        Initialize the downloader.
        
        Args:
            base_url: The directory URL to download from
            output_dir: Local directory to save files
            max_workers: Number of parallel downloads
            delay: Delay between requests (be nice to the server)
        """
        self.base_url = base_url.rstrip('/') + '/'
        self.output_dir = Path(output_dir)
        self.max_workers = max_workers
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def get_file_links(self, url=None):
        """
        Parse directory listing and extract all file/folder links.
        
        Returns:
            tuple: (files, subdirectories)
        """
        if url is None:
            url = self.base_url
            
        print(f"📂 Fetching directory listing: {url}")
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"❌ Error fetching {url}: {e}")
            return [], []
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        files = []
        subdirs = []
        
        # Find all links in the page
        for link in soup.find_all('a'):
            href = link.get('href')
            
            if not href:
                continue
                
            # Skip parent directory and sorting links
            if href in ['../', '../', '?', '#'] or href.startswith('?'):
                continue
            if 'Parent Directory' in link.text:
                continue
                
            full_url = urljoin(url, href)
            
            # Check if it's a subdirectory (ends with /)
            if href.endswith('/'):
                subdirs.append(full_url)
            else:
                # It's a file
                files.append(full_url)
        
        print(f"   Found {len(files)} files and {len(subdirs)} subdirectories")
        return files, subdirs
    
    def get_all_files_recursive(self, url=None, depth=0, max_depth=10):
        """
        Recursively get all files from directory and subdirectories.
        
        Args:
            url: Starting URL
            depth: Current recursion depth
            max_depth: Maximum recursion depth
            
        Returns:
            list: All file URLs
        """
        if depth > max_depth:
            print(f"⚠️ Max depth reached at {url}")
            return []
            
        if url is None:
            url = self.base_url
            
        all_files = []
        files, subdirs = self.get_file_links(url)
        all_files.extend(files)
        
        # Recursively get files from subdirectories
        for subdir in subdirs:
            time.sleep(self.delay)  # Be nice to the server
            all_files.extend(self.get_all_files_recursive(subdir, depth + 1, max_depth))
            
        return all_files
    
    def download_file(self, url, relative_path=None):
        """
        Download a single file.
        
        Args:
            url: File URL
            relative_path: Relative path to preserve directory structure
            
        Returns:
            tuple: (success, url, message)
        """
        try:
            # Determine local file path
            if relative_path:
                local_path = self.output_dir / relative_path
            else:
                # Extract path from URL relative to base
                parsed_base = urlparse(self.base_url)
                parsed_url = urlparse(url)
                relative = parsed_url.path.replace(parsed_base.path, '').lstrip('/')
                local_path = self.output_dir / relative
            
            # Create parent directories
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Skip if file already exists and has content
            if local_path.exists() and local_path.stat().st_size > 0:
                return (True, url, f"Skipped (exists): {local_path.name}")
            
            # Download the file
            response = self.session.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            # Get file size if available
            total_size = int(response.headers.get('content-length', 0))
            
            # Write to file
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            return (True, url, f"Downloaded: {local_path.name}")
            
        except Exception as e:
            return (False, url, f"Error: {str(e)}")
    
    def download_all(self, recursive=True, file_extensions=None):
        """
        Download all files from the directory.
        
        Args:
            recursive: Whether to download from subdirectories
            file_extensions: List of extensions to filter (e.g., ['.png', '.jpg'])
        """
        print(f"\n🚀 Starting download from: {self.base_url}")
        print(f"📁 Saving to: {self.output_dir.absolute()}\n")
        
        # Get all file URLs
        if recursive:
            all_files = self.get_all_files_recursive()
        else:
            all_files, _ = self.get_file_links()
        
        # Filter by extension if specified
        if file_extensions:
            file_extensions = [ext.lower() for ext in file_extensions]
            all_files = [f for f in all_files if any(f.lower().endswith(ext) for ext in file_extensions)]
        
        if not all_files:
            print("❌ No files found to download!")
            return
        
        print(f"\n📥 Downloading {len(all_files)} files...\n")
        
        # Download files with progress bar
        successful = 0
        failed = 0
        skipped = 0
        
        with tqdm(total=len(all_files), desc="Downloading", unit="file") as pbar:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all download tasks
                futures = {executor.submit(self.download_file, url): url for url in all_files}
                
                for future in as_completed(futures):
                    success, url, message = future.result()
                    
                    if success:
                        if "Skipped" in message:
                            skipped += 1
                        else:
                            successful += 1
                    else:
                        failed += 1
                        tqdm.write(f"❌ {message}")
                    
                    pbar.update(1)
                    time.sleep(self.delay / self.max_workers)  # Rate limiting
        
        # Summary
        print(f"\n{'='*50}")
        print(f"✅ Download Complete!")
        print(f"   • Downloaded: {successful}")
        print(f"   • Skipped (existing): {skipped}")
        print(f"   • Failed: {failed}")
        print(f"   • Total: {len(all_files)}")
        print(f"   • Location: {self.output_dir.absolute()}")
        print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Download all files from NSIDC/NOAA directory listings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all files from the default URL
  python nsidc_downloader.py
  
  # Download to a specific directory
  python nsidc_downloader.py --output ./my_data
  
  # Download only PNG files
  python nsidc_downloader.py --extensions .png
  
  # Download from a custom URL
  python nsidc_downloader.py --url https://noaadata.apps.nsidc.org/NOAA/G02135/north/daily/images/
  
  # Non-recursive download (only current directory)
  python nsidc_downloader.py --no-recursive
        """
    )
    
    parser.add_argument(
        '--url', '-u',
        default='https://noaadata.apps.nsidc.org/NOAA/G02135/south/daily/images/',
        help='URL of the directory to download from'
    )
    
    parser.add_argument(
        '--output', '-o',
        default='./nsidc_data',
        help='Output directory for downloaded files (default: ./nsidc_data)'
    )
    
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=3,
        help='Number of parallel downloads (default: 3, be nice to the server!)'
    )
    
    parser.add_argument(
        '--delay', '-d',
        type=float,
        default=0.5,
        help='Delay between requests in seconds (default: 0.5)'
    )
    
    parser.add_argument(
        '--extensions', '-e',
        nargs='+',
        default=None,
        help='File extensions to download (e.g., .png .jpg .gif)'
    )
    
    parser.add_argument(
        '--no-recursive',
        action='store_true',
        help='Only download files from the specified directory (no subdirectories)'
    )
    
    args = parser.parse_args()
    
    # Create downloader and start
    downloader = NSIDCDownloader(
        base_url=args.url,
        output_dir=args.output,
        max_workers=args.workers,
        delay=args.delay
    )
    
    downloader.download_all(
        recursive=not args.no_recursive,
        file_extensions=args.extensions
    )


if __name__ == '__main__':
    main()

    