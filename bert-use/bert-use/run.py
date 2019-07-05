import time

import bert_api

def main():
    t0=time.time()
    bert_api.classifyAutoLabels(
        'data/',
        'model/bert_config.json',
        'model/vocab.txt',
        'output/',
        do_train=True,
        do_eval=True,
        init_checkpoint='model/bert_model.ckpt')
    print('Time use: %f'%(time.time()-t0))

main()
