#提供API
#provide API

import os

import tensorflow as tf
import bert.run_classifier as brc
import bert

from .run_directly import CUserLabelTaskProcessor
from .label import makeLabelList

#contain codes from https://github.com/google-research/bert
def classify(
    data_dir:str,#The input data dir. Should contain the .tsv files (or other data files) 
    bert_config_file:str,#The config json file corresponding to the pre-trained BERT model.
    vocab_file:str,#The vocabulary file that the BERT model was trained on.
    output_dir:str,#The output directory where the model checkpoints will be written.

    #unnecessary parameters
    task_name:str='customer_task',
    labels:list=None,#a list of all labels
    init_checkpoint:str=None,#Initial checkpoint (usually from a pre-trained BERT model).
    do_lower_case:bool=True,#Whether to lower case the input text. 
    #Should be True for uncased models and False for cased models.

    max_seq_length:int=128,#The maximum total input sequence length after WordPiece tokenization. 
    #Sequences longer than this will be truncated, and sequences shorter than this will be padded.

    do_train:bool=False,#Whether to run training.
    do_eval:bool=False,#Whether to run eval on the dev set.
    do_predict:bool=False,#Whether to run the model in inference mode on the test set.
    train_batch_size:int=32,#Total batch size for training.
    eval_batch_size:int=8,#Total batch size for eval.
    predict_batch_size:int=5,#Total batch size for predict.
    learning_rate:float=5e-5,#The initial learning rate for Adam.
    num_train_epochs:float=3.0,#Total number of training epochs to perform.
    warmup_proportion:float=0.1,#Proportion of training to perform linear learning rate warmup for. 
    #E.g., 0.1 = 10% of training.

    save_checkpoints_steps:int=1000,#How often to save the model checkpoint.
    iterations_per_loop:int=1000,#How many steps to make in each estimator call.
    use_tpu:bool=False,#Whether to use TPU or GPU/CPU.
    tpu_name:str=None,#The Cloud TPU to use for training. This should be either the name 
    #used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.

    tpu_zone:str=None,#[Optional] GCE zone where the Cloud TPU is located in. If not 
    #specified, we will attempt to automatically detect the GCE project from metadata.

    gcp_project:str=None,#[Optional] Project name for the Cloud TPU-enabled project. If not 
    #specified, we will attempt to automatically detect the GCE project from metadata.

    master:str=None,#[Optional] TensorFlow master URL.
    num_tpu_cores:int=8#Only used if `use_tpu` is True. Total number of TPU cores to use.
    ):
    tf.logging.set_verbosity(tf.logging.INFO)
    processors={
        "cola": brc.ColaProcessor,
        "mnli": brc.MnliProcessor,
        "mrpc": brc.MrpcProcessor,
        "xnli": brc.XnliProcessor,
    }
    #设置参数do_lower_case和init_checkpoint
    bert.tokenization.validate_case_matches_checkpoint(do_lower_case,init_checkpoint)
    #参数do_train、do_eval和do_predict至少有其一为真
    if not do_train and not do_eval and not do_predict:
        raise ValueError("At least one of 'do_train', 'do_eval' or 'do_predict' must be True.")
    #加载BERT模型的配置文件
    bert_config=bert.modeling.BertConfig.from_json_file(bert_config_file)
    #最大序列长度
    if max_seq_length>bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model was only trained up to sequence length %d"%
            (max_seq_length,bert_config.max_position_embeddings))
    #确保输出文件夹存在
    tf.gfile.MakeDirs(output_dir)
    #根据任务名选择相应的处理类
    taskName=task_name.lower()
    if taskName=='customer_task':
        processor=CUserLabelTaskProcessor(labels)
    else:
        if taskName not in processors:
            raise ValueError('Task not found: %s'%(taskName))
        processor=processors[taskName]()
    label_list=processor.get_labels()
    #print(label_list)
    #分词器设置
    tokenizer=bert.tokenization.FullTokenizer(vocab_file=vocab_file,do_lower_case=do_lower_case)
    #显卡相关
    tpu_cluster_resolver=None
    if use_tpu and tpu_name:
        tpu_cluster_resolver=tf.contrib.cluster_resolver.TPUClusterResolver(tpu_name,zone=gcp_project)
    is_per_host=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config=tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=master,
        model_dir=output_dir,
        save_checkpoints_steps=save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=iterations_per_loop,
            num_shards=num_tpu_cores,
            per_host_input_for_training=is_per_host))
    #训练参数
    train_examples=None
    num_train_steps=None
    num_warmup_steps=None
    if do_train:
        train_examples=processor.get_train_examples(data_dir)
        num_train_steps=int(len(train_examples)/train_batch_size*num_train_epochs)
        num_warmup_steps=int(num_train_steps*warmup_proportion)
    model_fn=brc.model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=init_checkpoint,
        learning_rate=learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=use_tpu,
        use_one_hot_embeddings=use_tpu)
    #使用TPU
    # If TPU is not available, this will fall back to normal Estimator on CPU or GPU.
    estimator=tf.contrib.tpu.TPUEstimator(
        use_tpu=use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        predict_batch_size=predict_batch_size)
    if do_train:
        train_file=os.path.join(output_dir,"train.tf_record")
        brc.file_based_convert_examples_to_features(train_examples, label_list, max_seq_length, tokenizer, train_file)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn=brc.file_based_input_fn_builder(input_file=train_file,seq_length=max_seq_length,is_training=True,drop_remainder=True)
        estimator.train(input_fn=train_input_fn,max_steps=num_train_steps)
    if do_eval:
        eval_examples=processor.get_dev_examples(data_dir)
        num_actual_eval_examples=len(eval_examples)
        if use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on. These do NOT count towards the metric (all tf.metrics
            # support a per-instance weight, and these get a weight of 0.0).
            while len(eval_examples)%eval_batch_size!=0:
                eval_examples.append(brc.PaddingInputExample())
        eval_file=os.path.join(output_dir,"eval.tf_record")
        brc.file_based_convert_examples_to_features(eval_examples, label_list, max_seq_length, tokenizer, eval_file)
        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",len(eval_examples), num_actual_eval_examples,len(eval_examples) - num_actual_eval_examples)
        tf.logging.info("  Batch size = %d", eval_batch_size)
        # This tells the estimator to run through the entire set.
        eval_steps = None
        # However, if running eval on the TPU, you will need to specify the number of steps.
        if use_tpu:
            assert len(eval_example)%eval_batch_size==0
            eval_steps=int(len(eval_examples)//eval_batch_size)
        eval_drop_remainder = True if use_tpu else False
        eval_input_fn=brc.file_based_input_fn_builder(
            input_file=eval_file,seq_length=max_seq_length,is_training=False,drop_remainder=eval_drop_remainder)
        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
        output_eval_file = os.path.join(output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
    if do_predict:
        predict_examples = processor.get_test_examples(data_dir)
        num_actual_predict_examples = len(predict_examples)
        if use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on.
            while len(predict_examples) % predict_batch_size != 0:
                 predict_examples.append(brc.PaddingInputExample())
        predict_file = os.path.join(output_dir, "predict.tf_record")
        brc.file_based_convert_examples_to_features(predict_examples,label_list,max_seq_length,tokenizer,predict_file)
        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",len(predict_examples),
                       num_actual_predict_examples,len(predict_examples) - num_actual_predict_examples)
        tf.logging.info("  Batch size = %d", predict_batch_size)
        predict_drop_remainder = True if use_tpu else False
        predict_input_fn = brc.file_based_input_fn_builder(
            input_file=predict_file,seq_length=max_seq_length,is_training=False,drop_remainder=predict_drop_remainder)
        result = estimator.predict(input_fn=predict_input_fn)
        output_predict_file = os.path.join(output_dir, "test_results.tsv")
        with tf.gfile.GFile(output_predict_file, "w") as writer:
            num_written_lines = 0
            tf.logging.info("***** Predict results *****")
            for (i, prediction) in enumerate(result):
                probabilities = prediction["probabilities"]
                if i >= num_actual_predict_examples:
                    break
                output_line = "\t".join(str(class_probability)for class_probability in probabilities) + "\n"
                writer.write(output_line)
                num_written_lines += 1
        assert num_written_lines == num_actual_predict_examples

#a function to look for all labels in dataset and classify
def classifyAutoLabels(
    data_dir:str,#The input data dir. Should contain the .tsv files (or other data files) 
    bert_config_file:str,#The config json file corresponding to the pre-trained BERT model.
    vocab_file:str,#The vocabulary file that the BERT model was trained on.
    output_dir:str,#The output directory where the model checkpoints will be written.

    #unnecessary parameters
    task_name:str='customer_task',
    labels:list=None,#a list of all labels
    init_checkpoint:str=None,#Initial checkpoint (usually from a pre-trained BERT model).
    do_lower_case:bool=True,#Whether to lower case the input text. 
    #Should be True for uncased models and False for cased models.

    max_seq_length:int=128,#The maximum total input sequence length after WordPiece tokenization. 
    #Sequences longer than this will be truncated, and sequences shorter than this will be padded.

    do_train:bool=False,#Whether to run training.
    do_eval:bool=False,#Whether to run eval on the dev set.
    do_predict:bool=False,#Whether to run the model in inference mode on the test set.
    train_batch_size:int=32,#Total batch size for training.
    eval_batch_size:int=8,#Total batch size for eval.
    predict_batch_size:int=5,#Total batch size for predict.
    learning_rate:float=5e-5,#The initial learning rate for Adam.
    num_train_epochs:float=3.0,#Total number of training epochs to perform.
    warmup_proportion:float=0.1,#Proportion of training to perform linear learning rate warmup for. 
    #E.g., 0.1 = 10% of training.

    save_checkpoints_steps:int=1000,#How often to save the model checkpoint.
    iterations_per_loop:int=1000,#How many steps to make in each estimator call.
    use_tpu:bool=False,#Whether to use TPU or GPU/CPU.
    tpu_name:str=None,#The Cloud TPU to use for training. This should be either the name 
    #used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.

    tpu_zone:str=None,#[Optional] GCE zone where the Cloud TPU is located in. If not 
    #specified, we will attempt to automatically detect the GCE project from metadata.

    gcp_project:str=None,#[Optional] Project name for the Cloud TPU-enabled project. If not 
    #specified, we will attempt to automatically detect the GCE project from metadata.

    master:str=None,#[Optional] TensorFlow master URL.
    num_tpu_cores:int=8#Only used if `use_tpu` is True. Total number of TPU cores to use.
    ):
    if task_name=='customer_task' and labels is None:
        fileName=None
        if do_train:
            fileName='train.tsv'
        elif do_eval:
            fileName='dev.tsv'
        elif do_predict:
            fileName='test.tsv'
        else:
            raise ValueError("At least one of 'do_train', 'do_eval' or 'do_predict' must be True.")
        label_list=makeLabelList(os.path.join(data_dir,fileName))
    else:
        label_list=labels
    #print(label_list)
    classify(
        data_dir,
        bert_config_file,
        vocab_file,
        output_dir,
        task_name,
        label_list,
        init_checkpoint,
        do_lower_case,
        max_seq_length,
        do_train,
        do_eval,
        do_predict,
        train_batch_size,
        eval_batch_size,
        predict_batch_size,
        learning_rate,
        num_train_epochs,
        warmup_proportion,
        save_checkpoints_steps,
        iterations_per_loop,
        use_tpu,
        tpu_name,
        tpu_zone,
        gcp_project,
        master,
        num_tpu_cores)
