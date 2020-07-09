import dataHandler

testset = dataHandler.GenerateDatasets({"test":100})['test']
for i in range(len(testset)):
    print(i,testset[i])
