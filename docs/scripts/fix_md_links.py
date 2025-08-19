import os
import re

import requests

ultimate_replacements = [
    # Github video link, filename on RTD, download once boolean
    [
        "https://github.com/user-attachments/assets/a0a8bffb-bf81-4401-9ace-3b4955436b57",
        "waveorder-overview.mp4",
        False,
    ],
    [
        "https://github.com/user-attachments/assets/4f9969e5-94ce-4e08-9f30-68314a905db6",
        "figure-slideshow.mp4",
        False,
    ],
]


def fix_markdown_links(html_file):
    with open(html_file, "r+") as f:
        content = f.read()
        # Example: Replace relative paths that point to .md with .html
        content = re.sub(r"(href=\".*?).md", r"\1.html", content)
        # Example: Replace relative paths that point to -- with - for html targets
        content = re.sub(r"(?<=[\w])--(?=[\w])", r"-", content)
        # Replace : ultimate_replacements
        content = replace_github_videos(content)
        f.seek(0)
        f.write(content)
        f.truncate()


# <a class="github reference external" href="https://github.com/user-attachments/assets/a0a8bffb-bf81-4401-9ace-3b4955436b57">user-attachments/assets</a>
def replace_github_videos(content: str):
    pre_src = '<a class="github reference external" href="'
    post_src = '">user-attachments/assets</a>'
    pre_fin = '<video src="https://waveorder.readthedocs.io/en/latest/_static/videos/'
    post_fin = '" controls autoplay></video>'
    for replacements in ultimate_replacements:
        src_txt = pre_src + replacements[0] + post_src
        if src_txt in content:
            fin_txt = pre_fin + replacements[1] + post_fin
            content = content.replace(src_txt, fin_txt)
            print(f"Replacing '{src_txt}' with '{fin_txt}'")
            if not replacements[2]:
                success = download_video(replacements[0], replacements[1])
                if success:
                    replacements[2] = True
    return content


def download_video(src_url, filename):
    output_dir = os.environ.get("READTHEDOCS_OUTPUT", "_build/html")
    resp = requests.get(src_url)  # making requests to server
    full_mp4_path = os.path.join(output_dir, "html/_static/videos", filename)
    with open(
        full_mp4_path, "wb"
    ) as f:  # opening a file handler to create new file
        f.write(resp.content)  # writing content to file
        print(
            "File {src} downloaded to {dl}".format(
                src=src_url, dl=full_mp4_path
            )
        )
        return True
    return False


if __name__ == "__main__":
    output_dir = os.environ.get("READTHEDOCS_OUTPUT", "_build/html")
    try:
        full_videos_path = os.path.join(output_dir, "html/_static/videos/")
        # Create the directory and any missing parent directories
        os.makedirs(full_videos_path, exist_ok=True)
        print(f"Directory '{full_videos_path}' created successfully.")
    except OSError as e:
        print(f"Error creating directory: {e}")

    for root, _, files in os.walk(output_dir):
        for file in files:
            if file.endswith(".html"):
                fix_markdown_links(os.path.join(root, file))
