import argparse
from pathlib import Path
import shlex
from tqdm import tqdm
from subprocess import check_output

import numpy as np


def download(relative_path, filename):
    return f"""curl --header 'Host: stor-k.npcmr.ru' --user-agent 'Mozilla/5.0 (X11; Linux x86_64; rv:86.0) Gecko/20100101 Firefox/86.0' --header 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8' --header 'Accept-Language: en-US,en;q=0.5' --header 'Content-Type: application/x-www-form-urlencoded' --header 'Origin: https://stor-k.npcmr.ru' --header 'DNT: 1' --referer 'https://stor-k.npcmr.ru/fsdownload/JPFoO4IaB/COVID19_1110' --cookie 'sharing_sid=u8f7PNaYpykB-tMKqbmJ00jE2LGiPsPs' --header 'Upgrade-Insecure-Requests: 1' --request POST --data-urlencode 'dlname="masks.zip"' --data-urlencode 'path=["{relative_path}"]' --data-urlencode '_sharing_id="JPFoO4IaB"' --data-urlencode 'api=SYNO.FolderSharing.Download' --data-urlencode 'codepage=enu' --data-urlencode 'version=2' --data-urlencode 'method=download' --data-urlencode 'mode=download' 'https://stor-k.npcmr.ru/fsdownload/webapi/file_download.cgi/masks.zip' --output '{filename}'"""


def main(root):
    root = Path(root)

    ct0 = ['/COVID19_1110/studies/CT-0/' + f'study_{i:04d}.nii.gz' for i in range(1, 255)]
    ct1 = ['/COVID19_1110/studies/CT-1/' + f'study_{i:04d}.nii.gz' for i in range(255, 939)]
    ct2 = ['/COVID19_1110/studies/CT-2/' + f'study_{i:04d}.nii.gz' for i in range(939, 1064)]
    ct3 = ['/COVID19_1110/studies/CT-3/' + f'study_{i:04d}.nii.gz' for i in range(1064, 1109)]
    ct4 = ['/COVID19_1110/studies/CT-4/' + f'study_{i:04d}.nii.gz' for i in range(1109, 1111)]

    masks = ['/COVID19_1110/masks/' + f'study_{i:04d}_mask.nii.gz' for i in range(255, 305)]

    for foldername, relative_paths in [
        ('ct0', ct0), ('ct1', ct1), ('ct2', ct2), ('ct3', ct3), ('ct4', ct4), ('masks', masks)]:

        path = root / foldername
        path.mkdir(parents=True, exist_ok=True)

        for relative_path in tqdm(relative_paths):
            filename = relative_path.split('/')[-1]
            if (path / filename).exists():
                continue
            check_output(shlex.split(download(relative_path, filename)), cwd=path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output')
    args = parser.parse_args()

    main(args.output)
