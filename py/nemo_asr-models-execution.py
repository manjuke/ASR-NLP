#!/usr/bin/env python
# coding: utf-8

# In[16]:


import nemo
import nemo.collections.asr as nemo_asr

from jiwer import wer ##  libarry to compute WER
from statistics import mean  ## Used to compute mean WER
import pandas as pd #for IO

## change this base_path pointing to location of task_Data
base_path = '/home/manju/Desktop/assign/task_data/'

def readDatasetsForDecoding(csv_file):
    data = pd.read_csv(csv_file)
    wav_loc = data['path']
    ground_truth = data['transcription']
    #action = data['action']
    #object_category = data['object']
    #location = data['location']
    return wav_loc, ground_truth


def getAbsoluteWavPath(wav_loc):
    ##Add base path to wav_loc list to get absolute path
    files = []
    for fname in wav_loc:
        files.append(base_path + fname)
    #### files=files[0:2]
    return files


def decodeAndComputeWER(inputFileName, NemoModelName, outputFileName):
    print("*** Decoding wav_data of ", inputFileName, " and writing results to ", outputFileName, "*********")
    
    ## Reading the csv files
    wav_loc, ground_truth = readDatasetsForDecoding(inputFileName)
    files = getAbsoluteWavPath(wav_loc)
    
    index_cnt = 0
    wer_lst = []
    ## files=files[0:10]
    
    file_wr = open(outputFileName, 'w')
    #print(files)
    out = 'Wave-File-Name , ' + 'hypothesised-Transcription , ' + 'GroundTruth-Transcription' + 'WER' + '\n'
    file_wr.write(out)
    ## Transcribe validation data using NEMO model
    for fname, hypothesis in zip(files, NemoModelName.transcribe(paths2audio_files=files)):
        #print(f"{fname},\t \t \"{decoded_transption}\",\t \t \"{ground_truth[index_cnt]}\"")
        error = wer(ground_truth[index_cnt], hypothesis) ### compute WER
        wer_lst.append(error)
        ## append to write output
        out = '\"' + fname +'\", \"' + hypothesis + '\", \"' + ground_truth[index_cnt] + '\", \"' +  str(error) + '\"\n'
        ## Write to file
        file_wr.write(out)
        index_cnt += 1 ##increase the indent
    
 
    print("Results of decoding written to ", outputFileName, " file.")
    ## Compute Overall WER
    overall_WER = mean(wer_lst) * 100
    print("Overall WER is: ", round(overall_WER, 3), "%")
    print("Total number of files decoded is: ", index_cnt)
    
    out = '\"Overall WER is: ' + str(round(overall_WER, 3)) + "%, " +'\"Total number of files : "' + str(index_cnt) + '\n'
    file_wr.write(out)
    file_wr.close() ##close the file
    

## Read the data and put into different lists
def main(base_path):
    
    train_file = base_path + 'train_data.csv'
    valid_file = base_path + 'valid_data.csv'
    
    # Download various variants of Nemo models
    ## refer https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/results.html#english
    jasper_model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="stt_en_jasper10x5dr")
    quartznet_model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="QuartzNet15x5Base-En")
    ## conf_trandcr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name="stt_en_conformer_transducer_xxlarge")
    ## asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name="stt_en_conformer_ctc_xlarge")
    ## print(nemo_asr.models.EncDecCTCModelBPE.list_available_models())

    ## Derive hypothesis and compute WER
    ##for NemoModelName in jasper_model, quartznet_model:
    outputFileName = "jasper_validation_results.txt"
    decodeAndComputeWER(valid_file, jasper_model, outputFileName) 
    
    outputFileName = "quartznet_validation_results.txt"
    decodeAndComputeWER(valid_file, quartznet_model, outputFileName) 
    
    print("\n************** For reference : architecture of Jasper nemo model ************* \n")
    print(jasper_model)
    
    print("\n************** For reference : architecture of quartznet nemo model ************* \n")
    print(quartznet_model)
    


if __name__ == "__main__":
    main(base_path)


# In[ ]:




