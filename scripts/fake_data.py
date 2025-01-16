import numpy as np

features = np.random.randn(1000, 10)
labels = np.random.randint(0, 2, size=(1000, ))

np.save('../data/processed/fake_data.npy', {'X': features, 'Y': labels})
print('Fake data created at ../data/processed/fake_data.npy')