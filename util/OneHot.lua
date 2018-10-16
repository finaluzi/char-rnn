
local OneHot = torch.class('OneHot')

function OneHot:__init(outputSize, batSize)
  -- We'll construct one-hot encodings by using the index method to
  -- reshuffle the rows of an identity matrix. To avoid recreating
  -- it every iteration we'll cache it.
  self.outSize=outputSize
  self.batSize=batSize
  -- self._eye = torch.eye(outputSize):float()
  -- self.output:resize(batSize, outputSize)
end

function OneHot:initOutput()
	return torch.Tensor(self.batSize,self.outSize)
end

function OneHot:updateOutput(input,output)
	output:zero()
	for i=1,self.batSize do
		-- print(output[i],input)
		output[i][input[i]]=1
	end
	-- output:copy(self._eye:index(1, input:long()))
	-- print(output,input)
end

function OneHot:initUpdate(input)
	local output=self:initOutput()
	self:updateOutput(input,output)
	return output
end