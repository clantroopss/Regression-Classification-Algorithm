import os

# Keeping all the DATASETS in the folder: team_31

# Getting the current path in OS independent format
dir_path = os.path.dirname(os.path.realpath(__file__))

datasets = {
    'sum_without_noise': {
        'location': os.path.join(dir_path,'..','..','The SUM dataset','without noise','The SUM dataset, without noise.csv'),
        'header_present': True,
        'sep': ';',
        'classification_labels_present': True
    },
    'sum_with_noise': {
        'location': os.path.join(dir_path,'..','..','The SUM dataset','with noise','The SUM dataset, with noise.csv'),
        'header_present': True,
        'sep': ';',
        'classification_labels_present': True
    },
    'skin_nonskin': {
        'location': os.path.join(dir_path,'..','..','SkinNonSkin','Skin_NonSkin.txt'),
        'header_present': True,
        'sep': '	'
    },
    '3d_road_network': {
        'location': os.path.join(dir_path,'..','..','3D Road Network','3D_spatial_network.txt'),
        'header_present': False,
        'sep': ',',
        'classification_labels_present': False  # Use quantiles to get 3 classes again
    }
}
#list of chunk sizes
chunk_size = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000, 10000000, 50000000, 100000000]