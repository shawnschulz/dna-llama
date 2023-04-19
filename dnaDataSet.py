import torch
import polars as pl
from enformer_pytorch import Enformer, GenomeIntervalDataset
from datasets import concatenate_datasets, load_dataset
import os
import numpy as np
import pandas as pd
import pysam
import json 
import pickle
from llama_cpp import Llama

class dnaDataSet:
    def save(self, fp):
        '''
            save dnaDataSet as pickle somewhere
        '''
        file_name = fp
        with open(file_name, 'wb') as file:
            pickle.dump(self, file)
            print(f'dnaDataSet object successfully saved to "{file_name}"')
    
    def __init__(self, modelPath=False, memoryDir=os.getcwd()):
        self.mutationDictionary={}
        self.bamsDictionary={}
        self.tsv=pd.DataFrame()
        self.relativeContextLength=4
        self.memoryDir=memoryDir
        #promptOutput is a string formatted as a hf dataset
        self.promptOutput=''
        self.modelPath=modelPath
    
    def __getitem__(self, position):
        '''
        '''
        items=[
            self.mutationDictionary,
            self.bamsDictionary,
            self.tsv,
            self.relativeContextLength,
            self.memoryDir,
            #promptOutput is a string formatted as a hf dataset
            self.promptOutput
        ]
        return items[position]
    
    def __repr__(self):
        '''
            
        '''
        return f'dnaDataSet object\n current model being used: {str(self.modelPath)},\n mutationDictionary: {str(self.mutationDictionary)},\n bamsDictionary: {str(self.bamsDictionary)},\n tsv: {str(self.tsv)},\n relativeContextLength: {str(self.relativeContextLength)},\n memoryDir: {str(self.memoryDir)},\n promptOutput: {str(self.promptOutput)}'
    
    def __add__(self, other):
        '''
            returns a dnaDataset with consolidated mutationDictionary and bamsDictionary, however other info is kept from the first dictionary
        '''
        selfCopy = self
        otherCopy = other
        selfCopy.mutationDictionary.update(otherCopy.mutationDictionary)
        selfCopy.bamDictionary.update(otherCopy.mutationDictionary)
        return selfCopy
    
    def __len__(self):
        '''
            prints the length of the mutationDictioanry keys
        '''
        print("The length of the mutationDictionary keys is :")
        return(len(self.mutationDictionary))
    
    def setRelativeContextLength(self, contextLength):
        '''
            takes int contextLength and sets it in the dataset
        '''
        self.relativeContextLength=contextLength
    
    def saveOutput(self, fp, memoryDir=False):
        '''
            saves the output of a prompting to memoryDir by default (so it can be used automtically when calling prompting), but can also be called
            to save where user specifies filepath
        '''
        if not memoryDir:
            with open(fp, "w") as outfile:
                json.dump(self.promptOutput, outfile)
        else:
            ith open(self.memoryDir + '/' + fp, "w") as outfile:
                json.dump(self.promptOutput, outfile)
    
    def saveMutationDictionary(self, fp, memoryDir=False):
        '''
            saves mutationDictionary produced from tsv file and bam files to memoryDir by default as json file, but can also be called to save where user
            specifies filepath
        '''
        if not memoryDir:
            with open(fp, "w") as outfile:
                json.dump(self.mutationDictionary, outfile)
        else:
            with open(self.memoryDir + '/' + fp, "w") as outfile:
                json.dump(self.mutationDictionary, outfile)
    
    def makeLlamaDataset(self, tsv_dir, bam_path, bed_path):
        '''
            from a directory containing an annotated tsv file, many bam files and a bed path, create a huggingface dataset for use in llama

            start by just passing lines from vcf to llama for fine tuning, along with a line that says 
            "The read/basepairs/sequence at this position is:
            The read information from reference is:"

            This is a pretty brute force way to do it but maybe it'll create something coherent from llama.


            Getting correct sequence instruction: 
            "instruction": f"The gene {gene} is mutated at the {start_pos} basepair. What is the sequence? What is the mutation?",
            "input": f"{read_seq}",
            "output": "5"

            Getting whether exonic or not/amino acid change:


            [WIP] Instrucitons incorporating answers from databases:

            Clinvar:

            NCBI:

            Genecards: 

        '''
        for filename in os.listdir(tsv_dir):
            if filename.endswith('tsv'):
                tsv_file = os.path.join(tsv_dir, filename)
                tsv_length=len(tsv_file)
                counter = 0
                print("the tsv file is: ")
                print(tsv_file)
                for i in range(tsv_length):
                    chrom = tsv['CHROM'][i]
                    start_pos = tsv['POS'][i]
                    sample = tsv['SAMPLE'][i]
                    gene = tsv['Gene.refGene'][i]
                    gt = tsv['GT'][i]
                    alt = tsv['ALT'][i]
         #           print("tsv from the tsv file is: ")
          #          print(' '.join(tsv.columns))
                    if gt == '0/1' or gt == '1/1':
                        print(start_pos)
                        print(sample, gt)
                        print(alt)
                        print(gene)

                        ### position of mutation is the position is says on the pileup - start position (0 indexed)
                        ## start position can be greater than or less than position of read start, but luckily
                        ## should be able to index the base that's changed either way 

                        #

                        samfile = pysam.AlignmentFile(bam_path, "rb" )
                        self.bamsDictionary[bam_path] = samfile
                        pileup = samfile.pileup(chrom, start_pos, start_pos+1, min_mapping_quality=58)
                        for read in pileup:
                            read_list = str(read).split('\t')
                            read_start = read_list[5]
                            read_seq = read_list[11]

                            mutated_base= read_seq[int(read_start) - start_pos] 


                            print(f"the start pos from tsv is {start_pos} the start pos from pileup is {read_start} the the gene is: "+ gene +  ' the read is: ' + str(read_list) + ' and the mutated base is: ' + mutated_base)
                            print('for sanity, the mutated allele was: ' + alt)
                            self.mutationDictionary["Reference Genome: hg38, Read: " + read_seq] =  f"the start pos from tsv is {start_pos} the start pos from pileup is {read_start} the the gene is: "+ gene + ' and the mutated base is: ' + mutated_base
                    # for x in pileup:
                    #     if counter == 0:
                    #         print(str(x))

        return(mutation_dictionary)
    def fewShotLearning(self, read):
        '''
            takes a read as a prompt
            
            NOTE: currently only functional with a llama cpp model, may add functionality with other hf model calls when I actually have the gpu power to 
            do those lol
        '''
        counter = 0
        prompt_string = ''
        for key in self.mutationDictionary.keys():
            counter += 1
            if counter < self.relativeContextLength:
                prompt_string += "Input: " + key + "\n" + " Output: " + self.mutationDictionary[key] + "\n"
        prompt = 'Reference Genome: hg38, Read: ' + read
        output = llm(prompt_string + "\n" + "Input: " + prompt + "\n" + "Output: ", max_tokens=32, stop=["Input:"], echo=True)
        print(output)