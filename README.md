

# Project
This repository is the basic implementation and introduction for utilizing language models for reconstructing implicit knowledge, as described in our paper (Becker et al. 2021).

## Important Installation Requirements
PyTorch (conda environment)

`conda install pytorch torchvision cudatoolkit=10.1 -c pytorch `

Install tensorflow-datasets

`pip install tensorflow-datasets`

`pip install tensorflow-gpu`

Install transformers

`pip install transformers`

`pip install pytorch-lightning`

## Data examples
    Source Sentence 1: Not everyone should be obliged to pay the TV & radio licence.
    Source Sentence 2: Particularly the younger generations are no longer dependent on the programming of public broadcasters.    
    Concept1: public broadcasters
    Concept2: financed by the TV & radio licence    
    Path predicted by CONNECT: (public broadcasters, financed by the TV & radio licence)  -> receives action   
    Target Sentence: Public broadcasters are financed by the TV & radio licence.


## Models to be fine-tuned
[GPT-2](https://github.com/openai/gpt-2) | [XLNet](https://github.com/zihangdai/xlnet) | [BART](https://github.com/pytorch/fairseq/tree/master/examples/bart)

Our best performing language model BART, finetuned on e-SLNI (without constraints; with concepts as constraints; and with commonsense knowledge paths as constraints), can be downloaded from here:  https://drive.google.com/drive/folders/1FBiBlB_-V-6wgfIjUN_0JUn7VsEXO117


### Prepare training data (exclude the target sentence for test data in each line in case of GPT-2 and XLNet models)
    line = [sentence1, sentence2, concept1, concept2, target sentence]
    
    training: 
    GPT-2: 
            source line:  line[0]+'<|endoftext|>' + line[1] + '<|endoftext|>' + line[2] + '<sep>' + line[3] + '<sep>' + line[4] + '\n'
    XLNet:  
            source line: line[0]+'<sep>' + line[1] + '<sep>' + line[2] + '<sep>' + line[3] + '<sep>' + line[4] + '\n'
    BART:   
            source line: line[0]+'</s>' + line[1] + '</s>' + line[2] + '<sep>' + line[3]+ '\n'
            target line: line[4] +'\n'
    
    testing:
    GPT-2: 
            source line:  line[0]+'<|endoftext|>' + line[1] + '<|endoftext|>' + line[2] + '<sep>' + line[3] + '\n'
    XLNet:  
            source line: line[0]+'<sep>' + line[1] + '<sep>' + line[2] + '<sep>' + line[3] + '\n'
    BART:   
            source line: line[0]+'</s>' + line[1] + '</s>' + line[2] + '<sep>' + line[3]+ '\n'
    
    target line: line[4] +'\n'

### Write the prepared lines into files
    
    from preprocess import write_gpt2_file, write_xlnet_file, write_bart_file

    For example: 
        
        gpt2_path = 'data/gpt2/'
        file = 'train.source'
        write_gpt2_file(gpt2_train_lines, gpt2_path + file, mode ='train')
  
    Write the prepared training sources lines for GPT-2 and XLNet model with pad tokens (BART model pads the data itself during training): 
    
        from transformers import AutoTokenizer
        from preprocess import pad_sources
    
        gpt2_tokenizer = AutoTokenizer('gpt2')
        xlnet_tokenizer = AutoTokenizer('xlnet-large-cased')
  
        pad_sources(gpt2_tokenizer, new_path, gpt2_train_lines, block_size, model = 'gpt2')
        pad_sources(xlnet_tokenizer, new_path, xlnet_train_lines, block_size, model = 'xlnet')


## Fine-tuning
    GPT-2:
    python finetune_gpt2.py \
	--model_name_or_path=gpt2 \
	--model_type=gpt2 \
	--per_device_train_batch_size=8 \
	--per_gpu_train_batch_size=8 \
	--train_data_file=data/esnli/gpt2/train.source \
    --valid_data_file=data/esnli/gpt2/valid.source
	--output_dir=./finetune_gpt2_esnli \
	--do_train \
	--block_size=96 \
	--save_steps=500 \
	--save_total_limit=1 \

    XLNet: (the block_size has to be even)
    python finetune_xlnet.py \
	--model_name_or_path=xlnet-large-cased \
	--model_type=xlnet \
	--per_device_train_batch_size=8 \
	--per_gpu_train_batch_size=8 \
	--train_data_file=data/esnli/xlnet/train.source \
	--output_dir=./finetune_xlnet_esnli_heads_nl \
	--save_steps=500 \
	--block_size=96 \
	--save_total_limit=1 \
	--do_train
    
    BART: 
    python finetune_bart_pl.py \
	--model_name_or_path=facebook/bart-large-cnn \
	--tokenizer_name=facebook/bart-large-cnn \
	--learning_rate=3e-5 \
	--gpus=1 \
	--num_train_epochs=3 \
	--max_source_length=80 \
	--max_target_length=20 \
	--train_batch_size=8 \
	--data_dir=../data/esnli/bart/ \
	--output_dir=./finetune_bart_esnli \
	--do_train

## Generation
    Source lines should be prepared differently for each type of model:
    GPT-2: 
            source line:  line[0]+'<|endoftext|>' + line[1] + '<|endoftext|>' + line[2] + '<sep>' + line[3] + '\n'
    XLNet:  
            source line: line[0]+'<sep>' + line[1] + '<sep>' + line[2] + '<sep>' + line[3] + '\n'
    BART:   
            source line: line[0]+'</s>' + line[1] + '</s>' + line[2] + '<sep>' + line[3]+ '\n'


    Script: lm_generate.py

    The required arguments for running the generation script: 
        --model_path: where the fine-tuned model directory is stored
        --model_type: gpt2, xlnet or bart
        --test_src: the path to the test source file
        --save_path: where to save the generations

    For example:
    GPT-2:
    python lm_generate.py \
	        --model_path=finetune_gpt2_esnli_corec_path \
	        --model_type=gpt2 \
	        --test_src=data/ikat/ikat_test_corec_path.gpt2_source \
	        --save_path=data/esnli/ikat_test_corec_path.gpt2_pred
    
    XLNet:
    python lm_generate.py \
            --model_path=finetune_xlnet_esnli_corec_path \
            --model_type=xlnet \
	        --test_src=data/ikat/ikat_test_corec_path.xlnet_source \
	        --save_path=data/esnli/ikat_test_corec_path.xlnet_pred

    BART:
    python lm_generate.py \
	        --model_path=seq2seq/finetune_bart_esnli_corec_path/best_tfmr \
	        --model_type=bart \
	        --test_src=data/ikat/ikat_test_corec_path.bart_source \
	        --save_path=data/esnli/ikat_test_corec_path.bart_pred

### postprocess
    from postprocess import process_generations, write_generations
    
    For example:
    1. get the predicted lines from one prediction file: 
        path = 'data/esnli/ikat_test_corec_path.gpt2_pred'
        lines = [line.strip() for line in open(path).readlines()]
        generations = process_generations(lines, model_name='gpt2')

    2. postprocess and write the generations for the prediction files of one model:
        path = 'data/esnli/gpt2/'
        new_path = 'generations/esnli/gpt2/'
        write_generations(path, new_path, model_name = 'gpt2')
        

If you use our model, please cite:

Becker, M., Liang, S., and Frank, A. (2021c). Reconstructing Implicit Knowledge with Language Models. Accepted at: Deep Learning Inside Out (DeeLIO): Workshop on Knowledge Extraction and Integration for Deep Learning Architectures.

For questions or comments email us: mbecker@cl.uni-heidelberg.de










     

