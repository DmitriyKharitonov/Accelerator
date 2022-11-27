import os
import tensorflow as tf
from tensorflow import keras
import numpy as np


def main():
	model = keras.Sequential([
		keras.layers.Conv1D(10, kernel_size = 2, activation='relu', input_shape = (10,1)), 
	    keras.layers.MaxPooling1D(1),
	    keras.layers.Conv1D(32, 2, activation='relu'),
	    keras.layers.MaxPooling1D(2),
	    keras.layers.Conv1D(32, 2, activation='relu'),

	    keras.layers.Flatten(),
	    keras.layers.Dense(64, 'relu'),
	    keras.layers.Dense(2, 'softmax')
	])

	model.compile(loss="categorical_crossentropy", 
          			optimizer="adam",
          			metrics = ["accuracy"])

	my_data = np.genfromtxt(f'Data_no_negative.csv', delimiter = ',')
	y_train, x_train = my_data[:, 0].astype('int'), my_data[:, 1:11].astype('int')
	y_train_cat = keras.utils.to_categorical(y_train,2)
	x_train = x_train.reshape((x_train.shape[0], x_train.shape[1],1))
	# print(x_train.shape)

	model.fit(x_train, y_train_cat, epochs = 5)
	model.save(f'Model_no_negative.h5')


def test():
	my_data_test = np.genfromtxt(f'Test_no_negative.csv', delimiter = ',')
	y_test, x_test = my_data_test[:, 0].astype('int'), my_data_test[:, 1:11].astype('int')
	y_test_cat = keras.utils.to_categorical(y_test,2)
	model = keras.models.load_model('Model_no_negative.h5')
	x_test = x_test.reshape((x_test.shape[0], x_test.shape[1],1))

	results = model.evaluate(x_test, y_test_cat, batch_size=2)
	print('test loss, test acc:', results)


def loss_count():
	model = keras.models.load_model('Model(3).h5')
	my_data_test = np.genfromtxt(f'Test_new.csv', delimiter = ',')
	y_test, x_test = my_data_test[:, 0].astype('int'), my_data_test[:, 1:11].astype('int')
	y_test_cat = keras.utils.to_categorical(y_test,2)
	model = keras.models.load_model('Model(3).h5')

	#x_test = x_test.reshape((x_test.shape[0], x_test.shape[1],1))
	loss_count = 0
	all_data_lenght = x_test.shape[0]

	for i in range(all_data_lenght):
		current_x = np.reshape(x_test[i], (-1, len(x_test[i])))
		answ = np.argmax(model.predict(current_x.reshape((current_x.shape[0], current_x.shape[1],1))))
		if answ != y_test[i]:
			loss_count += 1
	print('loss count:', loss_count/all_data_lenght)


def predict():
	x_test = np.array([[13.3787308037281,63.5530098080635,52.7325958907604,18.8678183555603,56.0964588224888,53.038863837719,69.2329195439816,34.3681901693344,72.966315060854,70.6913800835609]])
	#print(x_test.shape[1])
	model = keras.models.load_model('Model(3).h5')
	x_test = x_test.reshape((x_test.shape[0], x_test.shape[1],1))
	print(model.predict(x_test))


if __name__ == '__main__':
	test()