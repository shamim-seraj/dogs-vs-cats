import test

if __name__ == '__main__':
    dog_sample = '/home/shuvornb/Desktop/Data Mining/dogs-vs-cats/test/21.jpg'
    cat_sample = '/home/shuvornb/Desktop/Data Mining/dogs-vs-cats/test/55.jpg'
    model_path = '/home/shuvornb/Desktop/Data Mining/dogs-vs-cats/dogs-vs-cats/saved_model'
    test.predict_sample(model_path, dog_sample)
