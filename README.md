# THESE ARE ALL WORKS IN PROGRESS - USE AND COMMENT ON AT YOUR OWN RISK

# dna-llama.ipynb
Few shot learning using alpaca lora quantized, parses annotated tsv files

To-do
- Format inputs to prompt for few shot learning
- Actually add the few shot learning using a LLM
- Add in option to input VCF and do vcf2tsv and annovar for the user

# dna-enformer
Take a fasta file containing many DNA artifacts and construct genomic tracks for the mutations, then compare to test files to find differences between
genomic tracks, ultimately using differences to remove artifacts

To-do
- Add in visualization to see whether it worked or not lol
- If it's working well, remove mutations with low loss

#dna-BERT
Similar idea to dna enformer but uses bert instead
