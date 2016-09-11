-- code adapted from https://github.com/jcjohnson/densecap

require 'cutorch'
require 'cunn'
require 'nnx'
require 'loadcaffe'

local net_utils = {}


function net_utils.subsequence(net, start_idx, end_idx)
--[[ Get a subsequence of a Sequential network.
Inputs:
  - net: A nn.Sequential instance
  - start_idx, end_idx: Start and end indices of the subsequence to get.
  Indices are inclusive.
Returns:
  - seq: A nn.Sequential network.
--]]
  local seq = nn.Sequential()
  for i = start_idx, end_idx do
    seq:add(net:get(i))
  end
  return seq
end


function net_utils.add_splits(net, start_idx, split_locs)
--[[ Recursively add split layers to a Sequential network using nn.ConcatTable. 
Useful for adding skip layers. 
Inputs:
  - net: A nn.Sequential instance
  - start_idx: Start index for this recursive call.
  - split_locs: Tensor of indices to add split at.
Returns:
  - model: A nn.Sequential module with skip layers.
--]]
	local model = nn.Sequential()
	-- Add everything before the split
	for i = start_idx, split_locs[1] do
		model:add(net:get(i))
	end
	-- Get the part after split
	local child
	if (#split_locs)[1] > 1 then
		-- If not last split, recurse
		child = net_utils.add_splits(net, 
			split_locs[1]+1, split_locs[{{2, (#split_locs)[1]}}])
	else
		-- If last split point, take remainder of net
		-- child = net_utils.subsequence(net, split_locs[1]+1, #net)
	end
	-- Add the child to the model
  if child ~= nil then
	 local children = nn.ConcatTable()
	 children:add(nn.Identity())
	 children:add(child)
   model:add(children)
  end
	return model
end


return net_utils


