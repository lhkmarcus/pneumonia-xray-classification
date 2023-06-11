import utils

if __name__ == "__main__":
    sort_obj = utils.ImageSorter(image_dir="./chest_xray")
    sort_obj.extract_images()
    sort_obj.sort_images()
    sort_obj.split_files()