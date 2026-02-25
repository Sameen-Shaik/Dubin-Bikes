import os
import urllib.request

URL = "https://data.smartdublin.ie/dataset/33ec9fe2-4957-4e9a-ab55-c5e917c7a9ab/resource/8b99c18f-bb0e-4f27-bdbb-c649f83dd487/download/dublinbike-historical-data-2023-06.csv"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(ROOT, "Data", "dataset.csv")

os.makedirs(os.path.dirname(DEST), exist_ok=True)

print(f"Downloading to {DEST} ...")
urllib.request.urlretrieve(URL, DEST)
print("Download complete.")
