import objaverse
import os
import numpy as np

import objaverse.xl as oxl

annotations = oxl.get_annotations()
print(annotations.keys())

download_list = annotations[annotations['source'] == 'sketchfab']
# download_list_wg = download_list[download_list['source']!='github']
print(download_list.head(10)['fileIdentifier'])
oxl.download_objects(download_list, 'data/objaverse', 
                     save_repo_format='files')