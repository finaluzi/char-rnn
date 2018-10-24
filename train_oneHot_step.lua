--[[

This file trains a character-level multi-layer RNN on text data

Code is based on implementation in
https://github.com/oxford-cs-ml-2015/practical6
but modified to have multi-layer support, GPU support, as well as
many other common model/optimization bells and whistles.
The practical6 code is in turn based on
https://github.com/wojciechz/learning_to_execute
which is turn based on other stuff in Torch, etc... (long lineage)

]] --

require "torch"
require "nn"
require "nngraph"
require "optim"
require "lfs"

require "util.OneHot"
require "util.misc"
local CharSplitLMMinibatchLoader = require "util.CharSplitLMMinibatchLoader"
local model_utils = require "util.model_utils"
local LSTM = require "model.LSTM_oneHot"

if not path then
    path = require "pl.path"
end

cmd = torch.CmdLine()
cmd:text()
cmd:text("Train a character-level language model")
cmd:text()
cmd:text("Options")
-- data
cmd:option("-data_dir", "data/tinyshakespeare", "data directory. Should contain the file input.txt with input data")
-- model params
cmd:option("-rnn_size", 300, "size of LSTM internal state")
cmd:option("-num_layers", 3, "number of layers in the LSTM")
cmd:option("-model", "lstm", "lstm,gru or rnn")
-- cmd:option('-emb_size', 100, 'embedding_size')
cmd:option("-seq_step", 1, "step of update c and h")

-- optimization
cmd:option("-learning_rate", 2e-3, "learning rate")
cmd:option("-learning_rate_decay", 0.97, "learning rate decay")
cmd:option("-learning_rate_decay_after", 10, "in number of epochs, when to start decaying the learning rate")
cmd:option("-decay_rate", 0.95, "decay rate for rmsprop")
cmd:option("-dropout", 0.5, "dropout for regularization, used after each RNN hidden layer. 0 = no dropout")
cmd:option("-seq_length", 32, "number of timesteps to unroll for")
cmd:option("-batch_size", 64, "number of sequences to train on in parallel")
cmd:option("-max_epochs", 1000, "number of full passes through the training data")
cmd:option("-grad_clip", 5, "clip gradients at this value")
cmd:option("-train_frac", 1, "fraction of data that goes into train set")
cmd:option("-val_frac", 0, "fraction of data that goes into validation set")
-- test_frac will be computed as (1 - train_frac - val_frac)
cmd:option("-init_from", "", "initialize network parameters from checkpoint at this path")
-- bookkeeping
cmd:option("-seed", 123, "torch manual random number generator seed")
cmd:option("-print_every", 10, "how many steps/minibatches between printing out the loss")
cmd:option("-eval_val_every", 1000, "every how many iterations should we evaluate on validation data?")
cmd:option("-checkpoint_dir", "cv", "output directory where checkpoints get written")
cmd:option("-savefile", "lstm", "filename to autosave the checkpont to. Will be inside checkpoint_dir/")
cmd:option(
    "-accurate_gpu_timing",
    0,
    "set this flag to 1 to get precise timings when using GPU. Might make code bit slower but reports accurate timings."
)
-- GPU/CPU
cmd:option("-gpuid", 0, "which gpu to use. -1 = use CPU")
cmd:option("-inputs", 1, "input1.txt, input2.txt")
cmd:option("-train_input", 1, "train input1.txt")
cmd:option("-fine_tune_start", 0, "turn off grad,  emb can <0")
cmd:option("-fine_tune_end", 0, "turn off grad,  dec can <0")
cmd:text()

-- parse input params
opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
-- train / val / test split for data, in fractions
local test_frac = math.max(0, 1 - (opt.train_frac + opt.val_frac))
local split_sizes = {opt.train_frac, opt.val_frac, test_frac}

-- initialize cunn/cutorch for training on the GPU and fall back to CPU gracefully
if opt.gpuid >= 0 then
    local ok, cunn = pcall(require, "cunn")
    local ok2, cutorch = pcall(require, "cutorch")
    if not ok then
        print("package cunn not found!")
    end
    if not ok2 then
        print("package cutorch not found!")
    end
    if ok and ok2 then
        print("using CUDA on GPU " .. opt.gpuid .. "...")
        cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        cutorch.manualSeed(opt.seed)
    else
        print("If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.")
        print("Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.")
        print("Falling back on CPU mode")
        opt.gpuid = -1 -- overwrite user setting
    end
end

-- create the data loader class
CharSplitLMMinibatchLoader.create_data(opt.data_dir, opt.inputs)
local loader =
    CharSplitLMMinibatchLoader.create_batch(opt.data_dir, opt.batch_size, opt.seq_length, split_sizes, opt.train_input)
local vocab_size = loader.vocab_size -- the number of distinct characters
local vocab = loader.vocab_mapping
print("vocab size: " .. vocab_size)

local one_hot = OneHot(vocab_size, opt.batch_size)

local real_x = {}
local real_x_temp = one_hot:initOutput()
for t = 1, opt.seq_length do
    real_x[t] = one_hot:initOutput()
end

-- make sure output directory exists
if not path.exists(opt.checkpoint_dir) then
    lfs.mkdir(opt.checkpoint_dir)
end

-- define the model: prototypes for one timestep, then clone them in time
local do_random_init = true
if string.len(opt.init_from) > 0 then
    print("loading an LSTM from checkpoint " .. opt.init_from)
    local checkpoint = torch.load(opt.init_from)
    protos = checkpoint.protos
    -- make sure the vocabs are the same
    local vocab_compatible = true
    for c, i in pairs(checkpoint.vocab) do
        if not vocab[c] == i then
            vocab_compatible = false
        end
    end
    assert(
        vocab_compatible,
        "error, the character vocabulary for this dataset and the one in the saved checkpoint are not the same. This is trouble."
    )
    -- overwrite model settings based on checkpoint to ensure compatibility
    print(
        "overwriting rnn_size=" ..
            checkpoint.opt.rnn_size .. ", num_layers=" .. checkpoint.opt.num_layers .. " based on the checkpoint."
    )
    opt.rnn_size = checkpoint.opt.rnn_size
    opt.num_layers = checkpoint.opt.num_layers
    do_random_init = false

    -- fine_tune!
    for i = 1, protos.rnn:size(1) do
        if protos.rnn:get(i).accGradParametersOrg then
            protos.rnn:get(i).accGradParameters = protos.rnn:get(i).accGradParametersOrg
            protos.rnn:get(i).accGradParametersOrg = nil
        end
    end
    local function dummyAccGradParameters()
    end
    if opt.fine_tune_start > 0 then
        for i = 1, opt.fine_tune_start do
            print(string.format("%d: %s", i, protos.rnn.modules[i]))
            protos.rnn:get(i).accGradParametersOrg = protos.rnn:get(i).accGradParameters
            protos.rnn:get(i).accGradParameters = dummyAccGradParameters
        end
    elseif opt.fine_tune_start < 0 then
        for i = 1, protos.rnn:size(1) + opt.fine_tune_start do
            print(string.format("%d: %s", i, protos.rnn.modules[i]))
            protos.rnn:get(i).accGradParametersOrg = protos.rnn:get(i).accGradParameters
            protos.rnn:get(i).accGradParameters = dummyAccGradParameters
        end
    end
    if opt.fine_tune_end > 0 then
        for i = opt.fine_tune_end + 1, protos.rnn:size(1) do
            print(string.format("%d: %s", i, protos.rnn.modules[i]))
            protos.rnn:get(i).accGradParametersOrg = protos.rnn:get(i).accGradParameters
            protos.rnn:get(i).accGradParameters = dummyAccGradParameters
        end
    elseif opt.fine_tune_end < 0 then
        for i = protos.rnn:size(1) + 1 + opt.fine_tune_end, protos.rnn:size(1) do
            print(string.format("%d: %s", i, protos.rnn.modules[i]))
            protos.rnn:get(i).accGradParametersOrg = protos.rnn:get(i).accGradParameters
            protos.rnn:get(i).accGradParameters = dummyAccGradParameters
        end
    end
else
    print("creating an " .. opt.model .. " with " .. opt.num_layers .. " layers")
    protos = {}
    if opt.model == "lstm" then
        protos.rnn = LSTM.lstm(vocab_size, opt.rnn_size, opt.num_layers, opt.dropout)
    end
    --graph.dot(protos.rnn.fg, 'Big MLP','rnn-link')

    --require 'sys'
    --sys.sleep(5000)

    protos.criterion = nn.ClassNLLCriterion()
end

-- the initial state of the cell/hidden states
init_state = {}
for L = 1, opt.num_layers do
    local h_init = torch.zeros(opt.batch_size, opt.rnn_size)
    if opt.gpuid >= 0 then
        h_init = h_init:cuda()
    end
    table.insert(init_state, h_init:clone())
    if opt.model == "lstm" then
        table.insert(init_state, h_init:clone())
    end
end

-- ship the model to the GPU if desired
if opt.gpuid >= 0 then
    for k, v in pairs(protos) do
        v:cuda()
    end
    for t = 1, opt.seq_length do
        real_x[t] = real_x[t]:cuda()
    end
    real_x_temp = real_x_temp:cuda()
end

-- put the above things into one flattened parameters tensor
params, grad_params = model_utils.combine_all_parameters(protos.rnn)

-- initialization
if do_random_init then
    params:uniform(-0.08, 0.08) -- small uniform numbers
end
-- initialize the LSTM forget gates with slightly higher biases to encourage remembering in the beginning
if opt.model == "lstm" then
    for layer_idx = 1, opt.num_layers do
        for _, node in ipairs(protos.rnn.forwardnodes) do
            if node.data.annotations.name == "i2h_" .. layer_idx then
                print("setting forget gate biases to 1 in LSTM layer " .. layer_idx)
                -- the gates are, in order, i,f,o,g, so f is the 2nd block of weights
                node.data.module.bias[{{opt.rnn_size + 1, 2 * opt.rnn_size}}]:fill(1.0)
            end
        end
    end
end

print("number of parameters in the model: " .. params:nElement())
-- make a bunch of clones after flattening, as that reallocates memory
clones = {}
for name, proto in pairs(protos) do
    print("cloning " .. name)
    -- clones[name] = model_utils.clone_many_times(proto, opt.seq_length, not proto.parameters)
    clones[name] = {}
end

-- preprocessing helper function
function prepro(x, y)
    x = x:transpose(1, 2):contiguous() -- swap the axes for faster indexing
    y = y:transpose(1, 2):contiguous()
    if opt.gpuid >= 0 then -- ship the input arrays to GPU
        -- have to convert to float because integers can't be cuda()'d
        x = x:float():cuda()
        y = y:float():cuda()
    end
    return x, y
end

-- evaluate the loss over an entire split
function eval_split(split_index, max_batches)
    print("evaluating loss over split index " .. split_index)
    local n = loader.split_sizes[split_index]
    if max_batches ~= nil then
        n = math.min(max_batches, n)
    end

    loader:reset_batch_pointer(split_index) -- move batch iteration pointer for this split to front
    local loss = 0

    local rnn_state_temp = init_state
    local prediction_temp = nil

    for i = 1, n do -- iterate over batches in the split
        -- fetch a batch
        local x, y = loader:next_batch(split_index)
        x, y = prepro(x, y)
        -- forward pass
        for t = 1, opt.seq_length do
            clones.rnn[t]:evaluate() -- for dropout proper functioning
            one_hot:updateOutput(x[t], real_x_temp)
            local lst = clones.rnn[t]:forward {real_x_temp, unpack(rnn_state_temp)}
            rnn_state_temp = {}
            for i = 1, #init_state do
                table.insert(rnn_state_temp, lst[i])
            end
            prediction_temp = lst[#lst]
            loss = loss + clones.criterion[t]:forward(prediction_temp, y[t])
        end
        -- carry over lstm state
        print(i .. "/" .. n .. "...")
    end

    loss = loss / opt.seq_length / n
    return loss
end

function init_clones(idx)
    for name, proto in pairs(protos) do
        if not clones[name][idx] then
            if idx % opt.print_every == 0 then
                print("cloned " .. name .. " " .. idx .. " / " .. opt.seq_length)
            end
            model_utils.clone_to_index(proto, name, idx, clones[name])
        end
    end
end

-- do fwd/bwd and return loss, grad_params
local init_state_global = clone_list(init_state)
local i_counter = opt.seq_step - 1

function feval(x)
    if x ~= params then
        params:copy(x)
    end
    grad_params:zero()

    ------------------ get minibatch -------------------
    local x, y = loader:next_batch(1)
    x, y = prepro(x, y)
    ------------------- forward pass -------------------
    local rnn_state = {[0] = init_state_global}
    local predictions = {} -- softmax outputs
    local loss = 0
    local lst = nil

    for t = 1, opt.seq_length do
        init_clones(t)
    end

    for t = 1, opt.seq_length do
        clones.rnn[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)
        one_hot:updateOutput(x[t], real_x_temp)
        lst = clones.rnn[t]:forward {real_x_temp, unpack(rnn_state[t - 1])}
        rnn_state[t] = {}
        for i = 1, #init_state do
            table.insert(rnn_state[t], lst[i])
        end -- extract the state, without output
        predictions[t] = lst[#lst]
        loss = loss + clones.criterion[t]:forward(predictions[t], y[t])
    end

    loss = loss / opt.seq_length

    ------------------ backward pass -------------------
    -- initialize gradient at time t to be zeros (there's no influence from future)
    local drnn_state = {[opt.seq_length] = clone_list(init_state, true)} -- true also zeros the clones
    for t = opt.seq_length, 1, -1 do
        -- backprop through loss, and softmax/linear
        local doutput_t = clones.criterion[t]:backward(predictions[t], y[t])
        table.insert(drnn_state[t], doutput_t)
        local dlst = clones.rnn[t]:backward({real_x[t], unpack(rnn_state[t - 1])}, drnn_state[t])
        drnn_state[t - 1] = {}
        for k, v in pairs(dlst) do
            if k > 1 then -- k == 1 is gradient on x, which we dont need
                -- note we do k-1 because first item is dembeddings, and then follow the
                -- derivatives of the state, starting at index 2. I know...
                drnn_state[t - 1][k - 1] = v
            end
        end
    end
    ------------------------ misc ----------------------
    -- for i=1,opt.seq_length do init_state_all[t]=rnn_state[t] end
    -- transfer final state to initial state (BPTT)
    i_counter = i_counter + 1
    if i_counter == opt.seq_step then
        i_counter = 0
        init_state_global = rnn_state[#rnn_state] -- NOTE: I don't think this needs to be a clone, right?
    end
    -- grad_params:div(opt.seq_length) -- this line should be here but since we use rmsprop it would have no effect. Removing for efficiency
    -- clip gradient element-wise
    grad_params:clamp(-opt.grad_clip, opt.grad_clip)
    return loss, grad_params
end

-- start optimization here
train_losses = {}
val_losses = {}
local optim_state = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
local iterations = opt.max_epochs * loader.ntrain
local iterations_per_epoch = loader.ntrain
local loss0 = nil
for i = 1, iterations do
    local epoch = i / loader.ntrain

    local timer = torch.Timer()
    --    local _, loss = optim.rmsprop(feval, params, optim_state)
    local _, loss = optim.adam(feval, params, optim_state)
    if opt.accurate_gpu_timing == 1 and opt.gpuid >= 0 then
        --[[
        Note on timing: The reported time can be off because the GPU is invoked async. If one
        wants to have exactly accurate timings one must call cutorch.synchronize() right here.
        I will avoid doing so by default because this can incur computational overhead.
        --]]
        cutorch.synchronize()
    end
    local time = timer:time().real

    local train_loss = loss[1] -- the loss is inside a list, pop it
    train_losses[i] = train_loss

    -- exponential learning rate decay
    if i % loader.ntrain == 0 and opt.learning_rate_decay < 1 then
        if epoch >= opt.learning_rate_decay_after then
            local decay_factor = opt.learning_rate_decay
            optim_state.learningRate = optim_state.learningRate * decay_factor -- decay it
            print("decayed learning rate by a factor " .. decay_factor .. " to " .. optim_state.learningRate)
        end
    end

    -- every now and then or on last iteration
    if i % opt.eval_val_every == 0 or i == iterations then
        -- evaluate loss on validation data
        local val_loss = eval_split(2) -- 2 = validation
        val_losses[i] = val_loss

        local savefile = string.format("%s/lm_%s_epoch%.2f_%.4f.t7", opt.checkpoint_dir, opt.savefile, epoch, val_loss)
        print("saving checkpoint to " .. savefile)
        local checkpoint = {}
        checkpoint.protos = protos
        checkpoint.opt = opt
        checkpoint.train_losses = train_losses
        checkpoint.val_loss = val_loss
        checkpoint.val_losses = val_losses
        checkpoint.i = i
        checkpoint.epoch = epoch
        checkpoint.vocab = loader.vocab_mapping
        checkpoint.vocab_size = vocab_size
        torch.save(savefile, checkpoint)
    end

    if i % opt.print_every == 0 then
        print(
            string.format(
                "%d/%d (epoch %.3f), train_loss = %6.8f, grad/param norm = %6.4e, time/batch = %.4fs",
                i,
                iterations,
                epoch,
                train_loss,
                grad_params:norm() / params:norm(),
                time
            )
        )
    end

    if i % 10 == 0 then
        collectgarbage()
    end

    -- handle early stopping if things are going really bad
    if loss[1] ~= loss[1] then
        print(
            "loss is NaN.  This usually indicates a bug.  Please check the issues page for existing issues, or create a new issue, if none exist.  Ideally, please state: your operating system, 32-bit/64-bit, your blas version, cpu/cuda/cl?"
        )
        break -- halt
    end
    if loss0 == nil then
        loss0 = loss[1]
    end
    if loss[1] > loss0 * 3 then
    -- print('loss is exploding, aborting.')
    -- break -- halt
    end
end
