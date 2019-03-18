def cuda_to_numpy(tensor):
    """Converts a cuda tensor to a numpy array in place
    Positional argument:
        tensor -- Tensor to convert to numpy array 
    """
    import torch

    if tensor.requires_grad:
        return tensor.detach().cpu().numpy()
    else:
        return tensor.cpu().numpy()

def load_data(bands, path):
	"""Load multispectral image data"""
	import gdal
	import numpy as np
	import os
	dirpath, dirname, filenames = next(os.walk(path))
	
	num_bands = 10

	data = []
	for file in filenames:
		bands = []
		image = gdal.Open(dirpath + "/" + file)
		
		for band_i in range(num_bands):
			rband = image.GetRasterBand(band_i + 1)
			# Fill nans in data
			bands.append(np.nan_to_num(rband.ReadAsArray()))
            
		data.append(bands)

	return np.array(data), filenames
	
def split_trainset(train_val_data, train_val_labels, ratio, batch_size):
	"""Split train data into train and validation set
	with given ratio"""
	import random
	from torch.utils.data import DataLoader
	
	data = list(zip(train_val_data, train_val_labels))

	random.shuffle(data)

	train_val_data, train_val_labels = list(zip(*data))

	train_ratio = ratio

	trainsize = int(len(train_val_data) * train_ratio)

	trainset = train_val_data[:trainsize]
	trainlabels = train_val_labels[:trainsize]

	valset = train_val_data[trainsize:]
	valabels = train_val_labels[trainsize:]
	
	trainset = list(zip(trainset, trainlabels))
	valset = list(zip(valset, valabels))
	
	trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
	val_loader = DataLoader(valset, batch_size, shuffle=True)
	
	return trainloader, val_loader
	
def merge_dims(image):
	"""Create an ndarray that merges the dimensions of the subarrays in 
	an ndarray"""
	import numpy as np
	
	shape = image.shape + image[0].shape
	new_image = np.ndarray(shape=shape)
	
	for i in range(image.shape[-1]):
		new_image[i] = image[i]

	return new_image

def chunk_image(data, k=3, label=False):
	"""
	Break a multispectral image into 9xkxk tensors
		image: Pytorch tensor with dimensions CxHxW where in this application
		C is channels, H is height, and W is width
	"""
	import numpy as np
	
	if label == True:
		try:
			height = data.shape[0]
			width = data.shape[1]
		except:
			print(data.shape)
	else:
		height = data.shape[1]
		width = data.shape[2]
	
	batches = (height - k + 1) * (width - k + 1)

	if label == True:
		shape = (batches)
	else:
		shape = (batches, 9, k, k)
		
	out = np.zeros(shape=shape)
	
	if label == True:
		offset = (k - 1) // 2
	
		out = data[offset:height - offset, offset:width - offset]
		out = np.ravel(out)
	else:
		batch = 0
		
		for h in range(0, height - 2):
			for w in range(0, width - 2):
				out[batch] = data[0:, h:h + 3, w:w + 3]
				batch += 1
    
	return out
	
def parallelize(model):
	"""
	Wrap pytorch model in layer to run on multiple GPUs
	"""
	import torch.cuda as cuda
	import torch.nn as nn
	
	device_ids = [i for i in range(cuda.device_count())]
	model = nn.DataParallel(model, device_ids=device_ids)
	return model
	

