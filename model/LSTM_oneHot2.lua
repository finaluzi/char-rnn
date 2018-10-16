local LSTM_oneHot2 = {}
function LSTM_oneHot2.lstm(input_size, rnn_size, n, dropout)
  dropout = dropout or 0 

  -- there will be 2*n+1 inputs
  local org_outs = {}
  
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prev_c[L]
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  local x, input_size_L
  local outputs = {}
  for L = 1,n do
    -- c,h from previos timesteps
    local prev_h = inputs[L*2+1]
    local prev_c = inputs[L*2]
    -- the input to this layer
    if L == 1 then 
	  x = inputs[1]
	  input_size_L = input_size
    else 
      x = outputs[(L-1)*2] 
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
      input_size_L = rnn_size
    end
    -- evaluate the input sums at once for efficiency
    local i2h = nn.Linear(input_size_L, 4 * rnn_size)(x):annotate{name='i2h_'..L}
    local h2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_h):annotate{name='h2h_'..L}
    local all_input_sums = nn.CAddTable()({i2h, h2h})

    local reshaped = nn.Reshape(4, rnn_size)(all_input_sums)
    local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
    -- decode the gates
    local in_gate = nn.Sigmoid()(n1)
    local forget_gate = nn.Sigmoid()(n2)
    local out_gate = nn.Sigmoid()(n3)
    -- decode the write inputs
    local in_transform = nn.Tanh()(n4)
    -- perform the LSTM update
    local next_c           = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
      })
    -- gated cells form the output
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
	table.insert(org_outs, next_h)
	
	if L>2 then
		-- print(L,#add_t)
		local all_h={next_h}
		if L<n then
			table.insert(all_h, org_outs[1])
		else
			for LL = 1,n-2 do
				table.insert(all_h, org_outs[LL])
			end
		end
		next_h = nn.CAddTable()(all_h)
	end
	
    table.insert(outputs, next_c)
    table.insert(outputs, next_h)
  end
  
  -- add out put
  -- local top_h = outputs[#outputs]
  -- if n>2 then
	-- local all_h={top_h}
	-- for L = 1,n-2 do
		-- table.insert(all_h, outputs[L*2])
	-- end
    -- top_h = nn.CAddTable()(all_h)
  -- end

  -- set up the decoder
  local top_h = outputs[#outputs]
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end
  local proj = nn.Linear(rnn_size, input_size)(top_h):annotate{name='decoder'}
  local logsoft = nn.LogSoftMax()(proj)
  table.insert(outputs, logsoft)

  return nn.gModule(inputs, outputs)
end


-- function LSTM.init_softmaxtree(input_size)
	-- local hierarchy = {}
	-- local root_id = 0
	-- local smt = nn.SoftMaxTree(input_size, hierarchy, root_id)
	-- return smt
-- end

return LSTM_oneHot2

