# THESE ARE ALL WORKS IN PROGRESS - USE AND COMMENT ON AT YOUR OWN RISK

Usage:

clone the repo:

```
git clone https://github.com/shawnschulz/dna-llama.git
cd dna-llama
```

install the requirements:

```
#right now need to install enformer and bert stuff on your own
pip install -r requirements.txt
```

In python, can import the dnaDataSet class by copying the python file to the directory with your notebook or script and adding this line to it:

```
#hope to make this a little less janky soon
from dnaDataSet import dnaDataSet
```

# dnaDataSet.py
Defines the dnaDataSet class, which is meant to be sort of like a scanpy for working with dna mutations with operability with llm datasets for finetuning and prompting with dna info from tsvs and vcfs. Currently just has a method to collect some dna dataset info from an annotated tsv and BAM files, and a method to do few shot learning with llama.cpp quantized models since thats the only thing I can perform inference on reliably at the moment. Also saves convos and outputs as json files.

To-do
- Modify to add option to specify few shot learning prompts via json file rather than producing automatically via tsvs and BAMs (might do this later)
- Add methods using langchain or pinecone api for saving and prompt convo info to vector database
- Add methods to use non llama.cpp models
- Add methods to fine tune non llama.cpp models
- Add method to perform vcf2tsv and annovar for user through python 

# dna-llama.ipynb
Few shot learning using alpaca lora 30B quantized, parses annotated tsv files

To-do
- Format inputs to prompt for few shot learning
- Add in option to input VCF and do vcf2tsv and annovar for the user

# dna-enformer
Take a fasta file containing many DNA artifacts and construct genomic tracks for the mutations, then compare to test files to find differences between
genomic tracks, ultimately using differences to remove artifacts

To-do
- Add in visualization to see whether it worked or not lol
- If it's working well, remove mutations with low loss

# dna-BERT
Similar idea to dna enformer but uses bert instead
