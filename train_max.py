import random
import subprocess
import sys

__author__ = 'uzi'


def exec_cmd1(data_dir, rnn_size, num_layers, seq_step, dropout, seq_length, batch_size, max_epochs, last_max_epochs, savefile, inputs, current, last, fine_tune_start, fine_tune_end):
    cmd_list = ['th', 'train_oneHot_step.lua', '-data_dir', 'data/'+data_dir, '-rnn_size', str(rnn_size), '-num_layers', str(num_layers), '-seq_step',
                str(seq_step), '-dropout', str(dropout), '-seq_length', str(
                    seq_length), '-batch_size', str(batch_size), '-max_epochs', str(max_epochs), '-savefile',
                savefile+'_'+str(current), '-inputs', str(inputs), '-train_input', str(
                    current), '-print_every',
                '10']
    if last > 0:
        cmd_list.extend(['-init_from', 'cv/lm_'+savefile+'_'+str(last) +
                         '_epoch'+str(last_max_epochs)+'.00_nan.t7', '-fine_tune_start', str(fine_tune_start), '-fine_tune_end', str(fine_tune_end)])
    subprocess.call(cmd_list)
    # print(fine_tune_start)


def get_fine_tune_start(all_layer, fine_tune_end):
    return random.randint(0, all_layer+fine_tune_end)


def get_train_idx(inputs):
    return random.randint(1, inputs)


def get_fine_tune_end(fine_tune_end):
    if random.random() > 0.5:
        return fine_tune_end
    return 0


if __name__ == '__main__':
    all_layer = int(sys.argv[1])
    data_dir = 'mix3'
    rnn_size = 800
    num_layers = 4
    seq_step = 3
    dropout = 0.5
    seq_length = 100
    max_epochs = 10
    savefile = data_dir+'_'+str(rnn_size)
    inputs = 5
    fine_tune_end = -3
    last_max_epochs = max_epochs
    current_idx = get_train_idx(inputs)
    batch_list = [48, 48, 1, 16, 48]
    last_idx = current_idx

    exec_cmd1(data_dir, rnn_size, num_layers, seq_step, dropout,
              seq_length, batch_list[current_idx-1], max_epochs, 0, savefile, inputs, current_idx, 0, 0, 0)
    loop = 50
    for i in range(loop):
        max_epochs = random.randint(1, 10)
        current_idx = get_train_idx(inputs)
        print('loop:', str(i), '/', str(loop))
        exec_cmd1(data_dir, rnn_size, num_layers, seq_step, dropout,
                  seq_length, batch_list[current_idx - 1], max_epochs,
                  last_max_epochs, savefile, inputs, current_idx, last_idx, get_fine_tune_start(all_layer, fine_tune_end), get_fine_tune_end(fine_tune_end))
        last_idx = current_idx
        last_max_epochs = max_epochs
