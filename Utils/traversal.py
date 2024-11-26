import os
from PIL import Image

def file_traversal(path):
    img_dir = []
    labels = []
    cnt = 0
    for device_name in os.listdir(path):
        device_path = os.path.join(path, device_name)
        for label in os.listdir(device_path):
            class_path = os.path.join(device_path, label)
            if label == 'Fake':
                for sub_dir in os.listdir(class_path):
                    sub_path = os.path.join(class_path, sub_dir)
                    for img in os.listdir(sub_path):
                        img_path = os.path.join(sub_path, img)
                        try:
                            img = Image.open(img_path)
                            img.convert('L')
                        except:
                            print('\rError image, skip to next')
                            print(img_path)
                        else:
                            img_dir.append(img_path)
                            labels.append(0)
                            cnt += 1
                            print(f'\rCollected {cnt} images', end='')
                        finally:
                            img.close()
            else:
                for img in os.listdir(class_path):
                    img_path = os.path.join(class_path, img)
                    try:
                        img = Image.open(img_path)
                        img.convert('L')
                    except:
                        print('\rError image, skip to next')
                        print(img_path)
                    else:
                        img_dir.append(img_path)
                        labels.append(1)
                        cnt += 1
                        print(f'\rCollected {cnt} images', end='')
                    finally:
                        img.close()

    return img_dir, labels