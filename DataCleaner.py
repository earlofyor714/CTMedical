import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

class DataCleaner:
    def __init__(self):
        self.a = 0

    def load_images(self):
        with np.load('data/ct-medical-image-analysis-tutorial/full_archive.npz') as img_data:
            full_image_dict = dict(zip(img_data['idx'], img_data['image']))
            return full_image_dict

    def load_features(self):
        overview_df = pd.read_csv('data/ct-medical-image-analysis-tutorial/overview.csv')
        overview_df.columns = ['idx'] + list(overview_df.columns[1:])
        overview_df['Contrast'] = overview_df['Contrast'].map(lambda x: 'Contrast' if x else 'No Contrast')
        return overview_df

    def data_overview(self):
        images = self.load_images()
        features = self.load_features()
        features['Age'].hist()
        plt.show()

        plt.matshow(images[0])
        plt.show()

        features['MeanHU'] = features['idx'].map(lambda x: np.mean(images.get(x, np.zeros((5, 12, 512)))))
        features['StdHU'] = features['idx'].map(lambda x: np.std(images.get(x, np.zeros((512, 512)))))
        print(features.sample(3))

        sns.set()
        _ = sns.pairplot(features[['Age', 'Contrast', 'MeanHU', 'StdHU']], hue="Contrast")
        sns.plt.show()


    def clean(self):
        images = self.load_images()
        features = self.load_features()
        y = features['Contrast']

        print('columns: {}'.format(features.columns.values))

        # to do: set output as contrast
        # to do: remove contrast from features
        # to do: normalize features (images as well?)
        # to do: combine features and images
        # finally: run random forest against it.  compare with CNN

dc = DataCleaner()
ov = dc.load_features()
dc.data_overview()
#print(ov.sample(3))
