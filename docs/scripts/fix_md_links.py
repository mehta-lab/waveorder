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
    [
        "https://user-images.githubusercontent.com/9554101/271128301-cc71da57-df6f-401b-a955-796750a96d88.mov",
        "2023-05-zebrafish.mov",
        False,
    ],
    [
        "https://user-images.githubusercontent.com/9554101/271128510-aa2180af-607f-4c0c-912c-c18dc4f29432.mp4",
        "2023-08-zebrafish-embryo.mp4",
        False,
    ],
    [
        "https://user-images.githubusercontent.com/9554101/273073475-70afb05a-1eb7-4019-9c42-af3e07bef723.mp4",
        "2023-10-05-recOrder-build-v2.mp4",
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
        # Replace : ultimate_replacements - github user-attachments
        content = replace_github_videos(content)
        # Replace : ultimate_replacements - githubusercontent
        content = replace_github_videos1(content)
        # Replace : ultimate_replacements - githubusercontent
        content = replace_github_videos2(content)
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
        if not replacements[2]:
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


# <a class="reference external" href="https://user-images.githubusercontent.com/9554101/271128301-cc71da57-df6f-401b-a955-796750a96d88.mov">https://user-images.githubusercontent.com/9554101/271128301-cc71da57-df6f-401b-a955-796750a96d88.mov</a>
def replace_github_videos1(content: str):
    pre_src = '<a class="reference external" href="'
    post_src1 = '">'
    post_src2 = "</a>"
    pre_fin = '<video src="https://waveorder.readthedocs.io/en/latest/_static/videos/'
    post_fin = '" controls autoplay></video>'
    for replacements in ultimate_replacements:
        if not replacements[2]:
            src_txt = (
                pre_src
                + replacements[0]
                + post_src1
                + replacements[0]
                + post_src2
            )
            if src_txt in content:
                fin_txt = pre_fin + replacements[1] + post_fin
                content = content.replace(src_txt, fin_txt)
                print(f"Replacing '{src_txt}' with '{fin_txt}'")
                if not replacements[2]:
                    success = download_video(replacements[0], replacements[1])
                    if success:
                        replacements[2] = True
    return content


# <video src="https://private-user-images.githubusercontent.com/9554101/271128301-cc71da57-df6f-401b-a955-796750a96d88.mov?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NTYwMjI2MDAsIm5iZiI6MTc1NjAyMjMwMCwicGF0aCI6Ii85NTU0MTAxLzI3MTEyODMwMS1jYzcxZGE1Ny1kZjZmLTQwMWItYTk1NS03OTY3NTBhOTZkODgubW92P1gtQW16LUFsZ29yaXRobT1BV1M0LUhNQUMtU0hBMjU2JlgtQW16LUNyZWRlbnRpYWw9QUtJQVZDT0RZTFNBNTNQUUs0WkElMkYyMDI1MDgyNCUyRnVzLWVhc3QtMSUyRnMzJTJGYXdzNF9yZXF1ZXN0JlgtQW16LURhdGU9MjAyNTA4MjRUMDc1ODIwWiZYLUFtei1FeHBpcmVzPTMwMCZYLUFtei1TaWduYXR1cmU9ZjkyNjNmNjk5YjM1OGMwMGQ0ZmRkYzQzZGIxMzQzOTYwMjk1NTFiZmI1YjAyZGVjZTQxODU3M2RhYWMxODc0MiZYLUFtei1TaWduZWRIZWFkZXJzPWhvc3QifQ.XTy2FO_rXMopZznWriS984hYel1l_MINmz27pceoxzU" data-canonical-src="https://private-user-images.githubusercontent.com/9554101/271128301-cc71da57-df6f-401b-a955-796750a96d88.mov?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NTYwMjI2MDAsIm5iZiI6MTc1NjAyMjMwMCwicGF0aCI6Ii85NTU0MTAxLzI3MTEyODMwMS1jYzcxZGE1Ny1kZjZmLTQwMWItYTk1NS03OTY3NTBhOTZkODgubW92P1gtQW16LUFsZ29yaXRobT1BV1M0LUhNQUMtU0hBMjU2JlgtQW16LUNyZWRlbnRpYWw9QUtJQVZDT0RZTFNBNTNQUUs0WkElMkYyMDI1MDgyNCUyRnVzLWVhc3QtMSUyRnMzJTJGYXdzNF9yZXF1ZXN0JlgtQW16LURhdGU9MjAyNTA4MjRUMDc1ODIwWiZYLUFtei1FeHBpcmVzPTMwMCZYLUFtei1TaWduYXR1cmU9ZjkyNjNmNjk5YjM1OGMwMGQ0ZmRkYzQzZGIxMzQzOTYwMjk1NTFiZmI1YjAyZGVjZTQxODU3M2RhYWMxODc0MiZYLUFtei1TaWduZWRIZWFkZXJzPWhvc3QifQ.XTy2FO_rXMopZznWriS984hYel1l_MINmz27pceoxzU" controls="controls" muted="muted" class="d-block rounded-bottom-2 border-top width-fit" style="max-height:640px; min-height: 200px"></video>
def replace_github_videos2(content: str):
    pre_fin = "https://waveorder.readthedocs.io/en/latest/_static/videos/"
    post_fin = ""
    for replacements in ultimate_replacements:
        if not replacements[2]:
            src_txt = replacements[0]
            vid_links = re.finditer(
                r"video src=\"(.*?)\"", content, re.MULTILINE
            )
            for vid_link in vid_links:
                if src_txt in content:
                    src_txt = vid_link
                    fin_txt = pre_fin + replacements[1] + post_fin
                    content = content.replace(src_txt, fin_txt)
                    print(f"Replacing '{src_txt}' with '{fin_txt}'")
                    if not replacements[2]:
                        success = download_video(
                            replacements[0], replacements[1]
                        )
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
