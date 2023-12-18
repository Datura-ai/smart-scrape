import requests
import re
import base64

def get_version(owner, repo, file_path, line_number):
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}"
    response = requests.get(url)
    if response.status_code == 200:
        content = response.json()['content']
        decoded_content = base64.b64decode(content).decode('utf-8')  # Decode from Base64
        lines = decoded_content.split('\n')
        if line_number <= len(lines):
            version_line = lines[line_number - 1]
            version_match = re.search(r'__version__ = "(.*?)"', version_line)
            if version_match:
                return version_match.group(1)
            else:
                raise Exception("Version information not found in the specified line")
        else:
            raise Exception("Line number exceeds file length")
    else:
        raise Exception("Failed to fetch file from GitHub")

try:
    version = get_version('corcel-api', 'cortex.t', 'template/__init__.py', 22)
    print(f"Version: {version}")
except Exception as e:
    print(f"Error: {e}")



