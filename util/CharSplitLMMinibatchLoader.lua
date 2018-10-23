-- Modified from https://github.com/oxford-cs-ml-2015/practical6
-- the modification included support for train/val/test splits

local CharSplitLMMinibatchLoader = {}
CharSplitLMMinibatchLoader.__index = CharSplitLMMinibatchLoader

function CharSplitLMMinibatchLoader.create_data(data_dir, max_idx)
    local vocab_file = path.join(data_dir, "vocab.t7")
    local tensor_file = path.join(data_dir, "data.t7")
    local run_prepro = false
    if not (path.exists(vocab_file) or path.exists(tensor_file)) then
        -- prepro files do not exist, generate them
        print("vocab.t7 and data.t7 do not exist. Running preprocessing...")
        run_prepro = true
    -- else
    -- check if the input file was modified since last time we
    -- ran the prepro. if so, we have to rerun the preprocessing
    -- local input_attr = lfs.attributes(input_file)
    -- local vocab_attr = lfs.attributes(vocab_file)
    -- local tensor_attr = lfs.attributes(tensor_file)
    -- if input_attr.modification > vocab_attr.modification or input_attr.modification > tensor_attr.modification then
    --     print("vocab.t7 or data.t7 detected as stale. Re-running preprocessing...")
    --     run_prepro = true
    -- end
    -- this thing what ever
    end
    if run_prepro then
        -- construct a tensor with all the data, and vocab file
        local tensor_list = {}
        local vocab_map = {}
        for idx = 1, max_idx, 1 do
            local input_file_num = "input" .. tostring(idx) .. ".txt"
            local input_file = path.join(data_dir, input_file_num)
            print("one-time setup: preprocessing input text file " .. input_file .. "...")
            CharSplitLMMinibatchLoader.text_to_tensor(input_file, vocab_map, tensor_list)
        end
        print("saving " .. vocab_file)
        torch.save(vocab_file, vocab_map)
        -- save output preprocessed files
        print("saving " .. tensor_file)
        torch.save(tensor_file, tensor_list)
    end
end

function CharSplitLMMinibatchLoader.create_batch(
    data_dir,
    batch_size,
    seq_length,
    split_fractions,
    file_idx)
    -- split_fractions is e.g. {0.9, 0.05, 0.05}
    local self = {}
    setmetatable(self, CharSplitLMMinibatchLoader)

    local vocab_file = path.join(data_dir, "vocab.t7")
    local tensor_file = path.join(data_dir, "data.t7")

    print("loading data files...")
    local data = torch.load(tensor_file)[file_idx]
    self.vocab_mapping = torch.load(vocab_file)

    -- cut off the end so that it divides evenly
    local len = data:size(1)
    if len % (batch_size * seq_length) ~= 0 then
        print("cutting off end of data so that the batches/sequences divide evenly")
        data = data:sub(1, batch_size * seq_length * math.floor(len / (batch_size * seq_length)))
    end

    -- count vocab
    self.vocab_size = 0
    for _ in pairs(self.vocab_mapping) do
        self.vocab_size = self.vocab_size + 1
    end

    -- self.batches is a table of tensors
    print("reshaping tensor...")
    self.batch_size = batch_size
    self.seq_length = seq_length

    local ydata = data:clone()
    ydata:sub(1, -2):copy(data:sub(2, -1))
    ydata[-1] = data[1]
    self.x_batches = data:view(batch_size, -1):split(seq_length, 2) -- #rows = #batches
    self.nbatches = #self.x_batches
    self.y_batches = ydata:view(batch_size, -1):split(seq_length, 2) -- #rows = #batches
    assert(#self.x_batches == #self.y_batches)

    -- lets try to be helpful here
    if self.nbatches < 50 then
        print(
            "WARNING: less than 50 batches in the data in total? Looks like very small dataset. You probably want to use smaller batch_size and/or seq_length."
        )
    end

    -- perform safety checks on split_fractions
    assert(
        split_fractions[1] >= 0 and split_fractions[1] <= 1,
        "bad split fraction " .. split_fractions[1] .. " for train, not between 0 and 1"
    )
    assert(
        split_fractions[2] >= 0 and split_fractions[2] <= 1,
        "bad split fraction " .. split_fractions[2] .. " for val, not between 0 and 1"
    )
    assert(
        split_fractions[3] >= 0 and split_fractions[3] <= 1,
        "bad split fraction " .. split_fractions[3] .. " for test, not between 0 and 1"
    )
    if split_fractions[3] == 0 then
        -- catch a common special case where the user might not want a test set
        self.ntrain = math.floor(self.nbatches * split_fractions[1])
        self.nval = self.nbatches - self.ntrain
        self.ntest = 0
    else
        -- divide data to train/val and allocate rest to test
        self.ntrain = math.floor(self.nbatches * split_fractions[1])
        self.nval = math.floor(self.nbatches * split_fractions[2])
        self.ntest = self.nbatches - self.nval - self.ntrain -- the rest goes to test (to ensure this adds up exactly)
    end

    self.split_sizes = {self.ntrain, self.nval, self.ntest}
    self.batch_ix = {0, 0, 0}

    print(
        string.format(
            "data load done. Number of data batches in train: %d, val: %d, test: %d",
            self.ntrain,
            self.nval,
            self.ntest
        )
    )
    collectgarbage()
    return self
end

function CharSplitLMMinibatchLoader:reset_batch_pointer(split_index, batch_index)
    batch_index = batch_index or 0
    self.batch_ix[split_index] = batch_index
end

function CharSplitLMMinibatchLoader:next_batch(split_index)
    if self.split_sizes[split_index] == 0 then
        -- perform a check here to make sure the user isn't screwing something up
        local split_names = {"train", "val", "test"}
        print("ERROR. Code requested a batch for split " .. split_names[split_index] .. ", but this split has no data.")
        os.exit() -- crash violently
    end
    -- split_index is integer: 1 = train, 2 = val, 3 = test
    self.batch_ix[split_index] = self.batch_ix[split_index] + 1
    if self.batch_ix[split_index] > self.split_sizes[split_index] then
        self.batch_ix[split_index] = 1 -- cycle around to beginning
    end
    -- pull out the correct next batch
    local ix = self.batch_ix[split_index]
    if split_index == 2 then
        ix = ix + self.ntrain
    end -- offset by train set size
    if split_index == 3 then
        ix = ix + self.ntrain + self.nval
    end -- offset by train + val
    return self.x_batches[ix], self.y_batches[ix]
end

function CharSplitLMMinibatchLoader.text_to_tensor(in_textfile_path, vocab_mapping, tensor_data)
    local timer = torch.Timer()

    print("loading text file...")
    local f = torch.DiskFile(in_textfile_path)
    local rawdata = f:readString("*a") -- NOTE: this reads the whole file at once
    f:close()
    local len = 0
    for _ in pairs(vocab_mapping) do
        len = len + 1
    end

    -- create vocabulary if it doesn't exist yet
    print("creating vocabulary mapping...")
    -- record all characters to a set
    local unordered = {}
    -- code snippets taken from http://lua-users.org/wiki/LuaUnicode
    for char in string.gfind(rawdata, "([%z\1-\127\194-\244][\128-\191]*)") do
        if not unordered[char] then
            unordered[char] = true
        end
        len = len + 1
    end
    -- sort into a table (i.e. keys become 1..N)
    local ordered = {}
    for char in pairs(unordered) do
        ordered[#ordered + 1] = char
    end
    table.sort(ordered)
    -- invert `ordered` to create the char->int mapping
    for i, char in ipairs(ordered) do
        vocab_mapping[char] = i
    end
    -- construct a tensor with all the data
    print("putting data into tensor...")
    local data = torch.ShortTensor(len) -- store it into 1D first, then rearrange
    local pos = 1
    for char in string.gfind(rawdata, "([%z\1-\127\194-\244][\128-\191]*)") do
        data[pos] = vocab_mapping[char]
        pos = pos + 1
    end
    tensor_data[#tensor_data + 1] = data
end

return CharSplitLMMinibatchLoader
