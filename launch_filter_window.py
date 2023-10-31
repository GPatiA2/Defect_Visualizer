from HSVFilterWindow import HSVFilterWindow


if __name__ == '__main__':

    image_path = 'dataset_generation/datadron_real'
    hsv_filter = HSVFilterWindow('image', image_path, (1920, 1080), 'dir', True, 'bad_pannels.json', True)
    hsv_filter.show_window()
    print('Done')