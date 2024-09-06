import zipfile
from tqdm import tqdm

def unzip_file(zip_file_path, output_folder="./dataset"):
  with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    total_files = len(zip_ref.infolist())
    with tqdm(total=total_files, desc="Extracting files") as pbar:
      for member in zip_ref.infolist():
        zip_ref.extract(member, output_folder)
        pbar.update(1)


if __name__ == '__main__':
  zip_file_path = "dataset.zip" 
  unzip_file(zip_file_path)
  print("Unzipped files to dataset folder.")