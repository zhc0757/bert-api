import os

import tensorflow as tf
import bert.run_classifier as brc
import bert

flags = tf.flags
FLAGS=flags.FLAGS

flags.DEFINE_string("label_file_path",None,"Please input label file path.")

#a processor to load your class
#reference from https://www.jiqizhixin.com/articles/2019-03-13-4
class CUserLabelTaskProcessor(brc.DataProcessor):
    labels=None

    def __init__(self,labels0):
        self.labels=labels0

    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        return self.labels

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = bert.tokenization.convert_to_unicode(line[1])
            label = bert.tokenization.convert_to_unicode(line[0])
            examples.append(brc.InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

class MyTaskProcessor(CUserLabelTaskProcessor):
    def __init__(self):
        if FLAGS.label_file_path is None:
            print('Label list cannot be none')
            super.__init__(self,[])
        else:
            f=open(FLAGS.label_file_path)
            tmpLabels=f.readlines()
            labels=[]
            for e in tmpLabels:
                labels.append(e.rstrip('\n'))
            super.__init__(self,labels)
            f.close()
            #print(self.labels)

#contain codes from https://github.com/google-research/bert
def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    processors={
        "cola": brc.ColaProcessor,
        "mnli": brc.MnliProcessor,
        "mrpc": brc.MrpcProcessor,
        "xnli": brc.XnliProcessor,
        "mytask":MyTaskProcessor
    }
    #设置参数do_lower_case和init_checkpoint
    bert.tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,FLAGS.init_checkpoint)
    #参数do_train、do_eval和do_predict至少有其一为真
    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError("At least one of 'do_train', 'do_eval' or 'do_predict' must be True.")
    #设置BERT模型的配置文件路径
    bert_config=bert.modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    #最大序列长度
    if FLAGS.max_seq_length>bert_config.max_position_embeddings:
        raise ValueError("Cannot use sequence length %d because the BERT model was only trained up to sequence length %d"%(FLAGS.max_seq_length,bert_config.max_position_embeddings))
    #确保输出文件夹存在
    tf.gfile.MakeDirs(FLAGS.output_dir)

    #根据任务名选择相应的处理类
    task_name=FLAGS.task_name.lower()
    if task_name not in processors:
        raise ValueError('Task not found: %s'%(task_name))
    processor=processors[task_name]()

    label_list=processor.get_labels()
    #分词器设置
    tokenizer=bert.tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file,do_lower_case=FLAGS.do_lower_case)

    #显卡相关
    tpu_cluster_resolver=None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver=tf.contrib.cluster_resolver.TPUClusterResolver(FLAGS.tpu_name,zone=FLAGS.gcp_project)
    is_per_host=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config=tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))
    
    #训练参数
    train_examples=None
    num_train_steps=None
    num_warmup_steps=None
    if FLAGS.do_train:
        train_examples=processor.get_train_examples(FLAGS.data_dir)
        num_train_steps=int(len(train_examples)/FLAGS.train_batch_size*FLAGS.num_train_epochs)
        num_warmup_steps=int(num_train_steps*FLAGS.warmup_proportion)
    model_fn=brc.model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    #使用TPU
    # If TPU is not available, this will fall back to normal Estimator on CPU or GPU.
    estimator=tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    if FLAGS.do_train:
        train_file=os.path.join(FLAGS.output_dir,"train.tf_record")
        brc.file_based_convert_examples_to_features(train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn=brc.file_based_input_fn_builder(input_file=train_file,seq_length=FLAGS.max_seq_length,is_training=True,drop_remainder=True)
        estimator.train(input_fn=train_input_fn,max_steps=num_train_steps)
    if FLAGS.do_eval:
        eval_examples=processor.get_dev_examples(FLAGS.data_dir)
        num_actual_eval_examples=len(eval_examples)
        if FLAGS.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on. These do NOT count towards the metric (all tf.metrics
            # support a per-instance weight, and these get a weight of 0.0).
            while len(eval_examples)%FLAGS.eval_batch_size!=0:
                eval_examples.append(brc.PaddingInputExample())
        eval_file=os.path.join(FLAGS.output_dir,"eval.tf_record")
        brc.file_based_convert_examples_to_features(eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)
        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",len(eval_examples), num_actual_eval_examples,len(eval_examples) - num_actual_eval_examples)
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)
        # This tells the estimator to run through the entire set.
        eval_steps = None
        # However, if running eval on the TPU, you will need to specify the number of steps.
        if FLAGS.use_tpu:
            assert len(eval_example)%FLAGS.eval_batch_size==0
            eval_steps=int(len(eval_examples)//FLAGS.eval_batch_size)
        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn=brc.file_based_input_fn_builder(input_file=eval_file,seq_length=FLAGS.max_seq_length,is_training=False,drop_remainder=eval_drop_remainder)
        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
    if FLAGS.do_predict:
        predict_examples = processor.get_test_examples(FLAGS.data_dir)
        num_actual_predict_examples = len(predict_examples)
        if FLAGS.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on.
            while len(predict_examples) % FLAGS.predict_batch_size != 0:
                 predict_examples.append(brc.PaddingInputExample())
        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        brc.file_based_convert_examples_to_features(predict_examples,label_list,FLAGS.max_seq_length,tokenizer,predict_file)
        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",len(predict_examples), num_actual_predict_examples,len(predict_examples) - num_actual_predict_examples)
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)
        predict_drop_remainder = True if FLAGS.use_tpu else False
        predict_input_fn = brc.file_based_input_fn_builder(input_file=predict_file,seq_length=FLAGS.max_seq_length,is_training=False,drop_remainder=predict_drop_remainder)
        result = estimator.predict(input_fn=predict_input_fn)
        output_predict_file = os.path.join(FLAGS.output_dir, "test_results.tsv")
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

def classify():
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()

if __name__=='__main__':
    classify()
