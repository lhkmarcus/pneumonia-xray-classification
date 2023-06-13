import os
import re
import sys
import shutil
import splitfolders

from src.logger import logging
from src.exceptions import CustomException

class ImageSorter:
    def __init__(self, image_dir):
        self.image_dir = image_dir

    def extract_images(self):
        logging.info("Started image extraction from subfolders.")
        if os.path.isdir(os.path.join(self.image_dir, "test", "NORMAL")):
            try:
                for root, _, files, in os.walk(self.image_dir, topdown=True):
                    for file in files:
                        if file.endswith((".png", ".jpg", ".jpeg")):
                            current_path = os.path.join(root, file)
                            shutil.move(current_path, self.image_dir)
                
                for folder in ("train", "test", "val"):
                    shutil.rmtree(os.path.join(self.image_dir, folder))
                logging.info("Finished extracting images.")

            except Exception as e:
                raise CustomException(e, sys)

    def sort_images(self):
        logging.info("Started image sorting into respective classes.")
        try:
            for pattern in ("normal", "bacteria", "virus"):
                dest_path = f"{self.image_dir}/{pattern}"
                if not os.path.exists(dest_path):
                    os.makedirs(dest_path)

                for file in os.listdir(self.image_dir):
                    if re.search(
                        f"({pattern}|{pattern.upper()}).*\.(png|jpg|jpeg)$", file):
                        shutil.move(f"{self.image_dir}/{file}", dest_path)

            dest_path = f"{self.image_dir}/normal"
            logging.info("Sorting residual images.")
            for file in os.listdir(self.image_dir):
                if re.search(f".*\.(png|jpg|jpeg)$", file):
                    shutil.move(f"{self.image_dir}/{file}", dest_path)
            logging.info("Finished sorting images.")
        
        except Exception as e:
            raise CustomException(e, sys)

    def split_files(self):
        try:
            splitfolders.ratio(
                self.image_dir, output="./artifacts", seed=42, ratio=(.8, .2), move=True)
            logging.info("Split files into training and testing sets.")

            for folder in ("bacteria", "normal", "virus"):
                shutil.rmtree(os.path.join(self.image_dir, folder))
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_image_sorting(self):
        self.extract_images()
        self.sort_images()
        self.split_files()

if __name__ == "__main__":
    ImageSorter(image_dir="./chest_xray").initiate_image_sorting()
