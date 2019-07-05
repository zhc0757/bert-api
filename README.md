# bert-api
a wrapper of bert-tensorflow

But so far, there are only APIs for classifier.

Assuming you have set up tensorflow or tensorflow-gpu, there are some steps to classify with BERT:

1. Install bert-api. `pip install bert-api`

2. Download and unzip model files from https://github.com/google-research/bert.

3. Prepare dataset. Dataset files are .tsv files in the same directory named train.tsv, dev.tsv or eval.tsv up to whether you want to train the model, evaluate the model or predict with the model. Every line in train.tsv and dev.tsv is as '(label)\t(text)'.

4. Program with Python3. Call function `bert_api.classify` or `bert_api.classifyAutoLabels`. The former needs a list of all labels in dataset. But the latter will look for all labels from dataset automatically so you do not need to make it by yourself. Whichever you use, you need to input other hyperparameters for the model while calling. Of course, you can use function `makeLabelList` to make a label list from dataset and save it for your usage.

5. Run your codes and wait.
