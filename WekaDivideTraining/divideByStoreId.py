## Divides training data into seperate .arff files by StoreId

import weka.core.jvm as jvm
from weka.core.converters import Loader, Saver
from weka.filters import Filter

def main():
	jvm.start()
	loader = Loader(classname="weka.core.converters.ArffLoader")
	data = loader.load_file("train_sorted.arff")
	numofStores = 1115

	for storeNum in range(0,numofStores):
	
		tempData = data
		removeUpper = Filter(classname="weka.filters.unsupervised.instance.RemoveWithValues",options=["-S",str(storeNum+2)+".0","-C","first","-L","first-last","-V"])		
		removeUpper.inputformat(data) 
		tempData = removeUpper.filter(data)

		removeLower = Filter(classname="weka.filters.unsupervised.instance.RemoveWithValues",options=["-S",str(storeNum+1)+".0","-C","first","-L","first-last"])		
		removeLower.inputformat(tempData) 
		tempData = removeLower.filter(tempData)
		
		#removing the storeID attribute
		tempData.delete_first_attribute()
		
		saver = Saver(classname="weka.core.converters.ArffSaver")
		saver.save_file(tempData, "stores/store"+str(storeNum+1)+".arff")   
		print 'Saved Store'+str(storeNum+1)
	

	jvm.stop()


if __name__ == '__main__':              
    main() 

