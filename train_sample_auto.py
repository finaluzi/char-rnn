import random
import subprocess

__author__ = 'uzi'


def exec_cmd1(data_dir, rnn_size, num_layers, seq_step, dropout, seq_length, batch_size, max_epochs, last_max_epochs, savefile, inputs, current, last, fine_tune_start, fine_tune_end, loop_idx):
    cmd_list = ['th', 'train_oneHot_step.lua', '-data_dir', 'data/'+data_dir, '-rnn_size', str(rnn_size), '-num_layers', str(num_layers), '-seq_step',
                str(seq_step), '-dropout', str(dropout), '-seq_length', str(
                    seq_length), '-batch_size', str(batch_size), '-max_epochs', str(max_epochs), '-savefile',
                savefile+'_'+str(current)+'_'+str(loop_idx), '-inputs', str(inputs), '-train_input', str(
                    current), '-print_every',
                '10']
    if last > 0:
        cmd_list.extend(['-init_from', 'cv/lm_'+savefile+'_'+str(last) + '_'+str(loop_idx-1) +
                         '_epoch'+str(last_max_epochs)+'.00_nan.t7', '-fine_tune_start', str(fine_tune_start), '-fine_tune_end', str(fine_tune_end)])
    subprocess.call(cmd_list)


def exec_sample(savefile, current, loop_idx, max_epochs):
    check_file = 'cv/lm_'+savefile+'_' + \
        str(current) + '_'+str(loop_idx)+'_epoch' + \
        str(max_epochs)+'.00_nan.t7'
    command = 'bash ./tangp2_oh.sh '+check_file + \
        ' 5000 李白 0 101 >> '+'output/'+savefile+'.txt'
    subprocess.run(command, shell=True, universal_newlines=True, check=True)


def exec_remove(savefile, last, loop_idx_1, last_max_epochs):
    check_file = 'cv/lm_'+savefile+'_' + \
        str(last) + '_'+str(loop_idx_1)+'_epoch' + \
        str(last_max_epochs)+'.00_nan.t7'
    command = 'rm '+check_file
    subprocess.run(command, shell=True, universal_newlines=True, check=True)


def get_fine_tune_start(all_layer):
    return random.randint(0, round(float(all_layer)/2.0))


def get_train_idx(inputs):
    return random.randint(1, inputs)


def get_fine_tune_end(all_layer):
    #  // floor division.
    return -random.randint(0, all_layer//2)


if __name__ == '__main__':
    all_layer = 75
    data_dir = 'mix3'
    rnn_size = 800
    num_layers = 4
    seq_step = 3
    dropout = 0.5
    seq_length = 100
    max_epochs = 3
    batch_list = [48, 48, 1, 4, 48]
    savefile = data_dir+'_'+str(rnn_size)
    inputs = len(batch_list)
    last_max_epochs = max_epochs
    current_idx = 1
    last_idx = current_idx

    exec_cmd1(data_dir, rnn_size, num_layers, seq_step, dropout,
              seq_length, batch_list[current_idx-1], max_epochs, 0, savefile, inputs, current_idx, 0, 0, 0, 0)
    i = 0
    while True:
        max_epochs = random.randint(1, 3)
        current_idx = get_train_idx(inputs)
        print('loop:', str(i), '/inf')
        exec_cmd1(data_dir, rnn_size, num_layers, seq_step, dropout,
                  seq_length, batch_list[current_idx - 1], max_epochs,
                  last_max_epochs, savefile, inputs, current_idx, last_idx, get_fine_tune_start(all_layer), get_fine_tune_end(all_layer), i+1)
        exec_remove(savefile, last_idx, i, last_max_epochs)
        exec_sample(savefile, current_idx, i+1, max_epochs)
        last_idx = current_idx
        last_max_epochs = max_epochs
        i += 1
