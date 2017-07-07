import glob

car_file_names = ["vehicles/GTI_Far/*.png", "vehicles/GTI_Left/*.png", "vehicles/GTI_MiddleClose/*.png", "vehicles/GTI_Right/*.png", "vehicles/KITTI_extracted/*.png"]
non_car_file_names = ["non-vehicles/Extras/*.png", "non-vehicles/GTI/*.png"]

def get_car_names():
    cars = []
    for file in car_file_names:
        cars += glob.glob(file)
    return cars

def get_non_car_names():
    not_cars = []
    for file in non_car_file_names:
        not_cars += glob.glob(file)
    return not_cars