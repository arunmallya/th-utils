-- code adapted from https://github.com/jcjohnson/densecap
require 'cutorch'
require 'cunn'
require 'nnx'
require 'loadcaffe'

local net_utils = {}

--[[
Load a cnn using loadcaffe.
Inputs:
- name: CNN name
- backend: loadcaffe backend to use - nn, cudnn, etc.
- path_offset: path to cnn model
--]]
function net_utils.load_cnn(name, backend, path_offset)
  local model_dir, proto_file, model_file, layer2index = nil, nil, nil, nil
  if name == 'fcn-32s-nyud-rgb' then
    model_dir = 'data/models/fcn-32s-nyud'
    proto_file = 'fcn-32s-nyud-rgb.prototxt'
    model_file = 'fcn-32s-nyud-rgb.caffemodel'
    -- Layer indices after relu
    layer2index = {conv1_2=4, conv2_2=9, conv3_3=16, conv4_3=23, conv5_3=30}
    layer_dim = {conv1_2=64, conv2_2=128, conv3_3=256, conv4_3=512, conv5_3=512, fc7=4096}
  elseif name == 'vgg-16' then
    model_dir = 'data/models/vgg-16'
    proto_file = 'VGG_ILSVRC_16_layers_deploy.prototxt'
    model_file = 'VGG_ILSVRC_16_layers.caffemodel'
    -- Layer indices after relu
    layer2index = {conv1_2=4, conv2_2=9, conv3_3=16, conv4_3=23, conv5_3=30}
    layer_dim = {conv1_2=64, conv2_2=128, conv3_3=256, conv4_3=512, conv5_3=512, fc7=4096}
  else
    error(string.format('Unrecognized model "%s"', name))
  end
  if path_offset then
    model_dir = paths.concat(path_offset, model_dir)
  end
  print('loading network weights from .. ' .. model_file)
  proto_file = paths.concat(model_dir, proto_file)
  model_file = paths.concat(model_dir, model_file)
  local cnn = loadcaffe.load(proto_file, model_file, backend)
  return {cnn, layer2index, layer_dim}
end


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


